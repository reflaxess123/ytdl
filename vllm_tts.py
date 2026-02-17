#!/usr/bin/env python3
"""TTS генерация через vLLM-Omni (быстрее чем HuggingFace)"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
import time
import os

# Настройки
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CACHE_DIR = Path("/app/downloads/.cache_5_Simple_Steps_for_S_f5d99c9c")
REFERENCE_AUDIO = "/app/downloads/.cache_5_Simple_Steps_for_S_f5d99c9c/reference.wav"
STAGE_CONFIGS_PATH = "/app/qwen3_tts_stage.yaml"


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


def main():
    print("=" * 60)
    print("vLLM-Omni TTS Generation")
    print("=" * 60)

    # Импорты vLLM
    from vllm import SamplingParams
    from vllm_omni import Omni

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

    # Инициализируем vLLM-Omni
    print(f"\nЗагрузка модели {TTS_MODEL}...")
    print(f"Stage configs: {STAGE_CONFIGS_PATH}")
    omni = Omni(
        model=TTS_MODEL,
        stage_configs_path=STAGE_CONFIGS_PATH,
        stage_init_timeout=600,  # 10 минут на загрузку модели
    )
    print("Модель загружена")

    # Параметры генерации
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        seed=42,
        detokenize=False
    )

    total_time = 0

    for idx, grp_idx in enumerate(groups_to_generate):
        grp = merged_segments[grp_idx]
        text = grp["translated"]
        dur = grp["end"] - grp["start"]

        print(f"\n[{idx+1}/{len(groups_to_generate)}] G{grp_idx}: {len(text)} chars, {dur:.1f}s")
        print(f"  Текст: {text[:80]}...")

        t0 = time.time()

        try:
            # Формируем запрос для voice clone
            inputs = {
                "additional_information": {
                    "task_type": ["Base"],
                    "ref_audio": [REFERENCE_AUDIO],
                    "ref_text": [""],  # x_vector_only_mode
                    "text": [text],
                    "language": ["Russian"],
                    "x_vector_only_mode": [True],
                    "max_new_tokens": [4096],
                }
            }

            # Генерация
            omni_generator = omni.generate([inputs], [sampling_params])

            for stage_outputs in omni_generator:
                for output in stage_outputs.request_output:
                    audio_tensor = output.multimodal_output["audio"]
                    sample_rate = output.multimodal_output["sr"].item()

                    # Конвертируем в numpy
                    if hasattr(audio_tensor, 'cpu'):
                        wav = audio_tensor.cpu().numpy()
                    else:
                        wav = np.array(audio_tensor)

            gen_time = time.time() - t0
            total_time += gen_time
            audio_dur = len(wav) / sample_rate
            rtf = gen_time / audio_dur

            # Сохраняем
            npy_file = output_dir / f"group_{grp_idx:04d}.npy"
            wav_file = output_dir / f"group_{grp_idx:04d}.wav"
            np.save(npy_file, wav)
            sf.write(str(wav_file), wav, sample_rate)

            print(f"  Готово: {gen_time:.1f}s, аудио: {audio_dur:.1f}s, RTF: {rtf:.2f}x")
            print(f"  Сохранено: {wav_file.name}")

        except Exception as e:
            print(f"  ОШИБКА: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Готово! Общее время: {total_time:.1f}s ({total_time/60:.1f} мин)")


if __name__ == "__main__":
    main()
