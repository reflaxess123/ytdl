#!/usr/bin/env python3
"""Параллельная генерация TTS с несколькими воркерами"""

import json
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import time
import multiprocessing as mp
from typing import List

# Настройки
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CACHE_DIR = Path("downloads/.cache_5_Simple_Steps_for_S_f5d99c9c")
REFERENCE_AUDIO = "D:/temp/test.wav"
NUM_WORKERS = 1  # 1 воркер - самый быстрый на одном GPU


def merge_segments_by_pause(segments, max_pause=0.5, min_duration=10.0, max_duration=60.0):
    """Объединяет сегменты по паузам"""
    if not segments:
        return []

    groups = []
    current_group = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "translated": segments[0].get("translated", ""),
        "original_indices": [0]
    }

    for i in range(1, len(segments)):
        seg = segments[i]
        pause = seg["start"] - current_group["end"]
        new_duration = seg["end"] - current_group["start"]

        if pause <= max_pause and new_duration <= max_duration:
            current_group["end"] = seg["end"]
            current_group["translated"] += " " + seg.get("translated", "")
            current_group["original_indices"].append(i)
        else:
            if current_group["end"] - current_group["start"] < min_duration and groups:
                prev = groups[-1]
                prev["end"] = current_group["end"]
                prev["translated"] += " " + current_group["translated"]
                prev["original_indices"].extend(current_group["original_indices"])
            else:
                groups.append(current_group)

            current_group = {
                "start": seg["start"],
                "end": seg["end"],
                "translated": seg.get("translated", ""),
                "original_indices": [i]
            }

    if current_group["end"] - current_group["start"] < min_duration and groups:
        prev = groups[-1]
        prev["end"] = current_group["end"]
        prev["translated"] += " " + current_group["translated"]
        prev["original_indices"].extend(current_group["original_indices"])
    else:
        groups.append(current_group)

    return groups


def worker_process(worker_id: int, group_indices: List[int], merged_segments: List[dict],
                   output_dir: Path, reference_audio: str):
    """Воркер - загружает модель и генерирует свои группы"""
    print(f"[Worker {worker_id}] Запуск, групп: {len(group_indices)}")

    from qwen_tts import Qwen3TTSModel
    import warnings
    warnings.filterwarnings("ignore", message=".*pad_token_id.*")

    device = "cuda:0"
    dtype = torch.bfloat16

    print(f"[Worker {worker_id}] Загрузка модели...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            TTS_MODEL,
            device_map=device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
    except:
        model = Qwen3TTSModel.from_pretrained(
            TTS_MODEL, device_map=device, dtype=dtype,
        )

    print(f"[Worker {worker_id}] Модель загружена, начинаю генерацию")

    for idx, grp_idx in enumerate(group_indices):
        grp = merged_segments[grp_idx]
        text = grp["translated"]
        dur = grp["end"] - grp["start"]

        print(f"[Worker {worker_id}] [{idx+1}/{len(group_indices)}] G{grp_idx}: {len(text)} chars, {dur:.1f}s")

        t0 = time.time()
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="Russian",
                ref_audio=reference_audio,
                x_vector_only_mode=True,
            )
            wav = wavs[0]
            gen_time = time.time() - t0

            # Сохраняем
            npy_file = output_dir / f"group_{grp_idx:04d}.npy"
            wav_file = output_dir / f"group_{grp_idx:04d}.wav"
            np.save(npy_file, wav)
            sf.write(str(wav_file), wav, sr)

            print(f"[Worker {worker_id}] [{idx+1}/{len(group_indices)}] G{grp_idx}: готово за {gen_time:.1f}s -> {wav_file.name}")

        except Exception as e:
            print(f"[Worker {worker_id}] G{grp_idx}: ОШИБКА {e}")
            wav = np.zeros(1000, dtype=np.float32)
            np.save(output_dir / f"group_{grp_idx:04d}.npy", wav)

    print(f"[Worker {worker_id}] Завершён")
    del model
    torch.cuda.empty_cache()


def main():
    print("="*60)
    print("Параллельная генерация TTS")
    print("="*60)

    # Загружаем перевод
    translated_file = CACHE_DIR / "translated.json"
    with open(translated_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Объединяем в группы
    merged_segments = merge_segments_by_pause(segments)
    total_groups = len(merged_segments)
    print(f"Всего групп: {total_groups}")

    # Папка для результатов
    output_dir = CACHE_DIR / "tts_merged"
    output_dir.mkdir(exist_ok=True)

    # Определяем какие группы уже готовы
    groups_to_generate = []
    for i in range(total_groups):
        npy_file = output_dir / f"group_{i:04d}.npy"
        if not npy_file.exists():
            groups_to_generate.append(i)

    print(f"Уже готово: {total_groups - len(groups_to_generate)}")
    print(f"Осталось: {len(groups_to_generate)}")

    if not groups_to_generate:
        print("Все группы готовы!")
        return

    # Распределяем по воркерам
    num_workers = min(NUM_WORKERS, len(groups_to_generate))
    chunks = [[] for _ in range(num_workers)]
    for i, grp_idx in enumerate(groups_to_generate):
        chunks[i % num_workers].append(grp_idx)

    print(f"\nВоркеров: {num_workers}")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: {len(chunk)} групп - {chunk[:5]}{'...' if len(chunk) > 5 else ''}")

    print("\nЗапуск воркеров...")
    t0 = time.time()

    # Запускаем процессы
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, chunks[worker_id], merged_segments, output_dir, REFERENCE_AUDIO)
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # Небольшая задержка между запусками чтобы не перегружать GPU

    # Ждём завершения
    for p in processes:
        p.join()

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Готово! Общее время: {total_time:.1f}s ({total_time/60:.1f} мин)")
    print(f"Файлы: {output_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
