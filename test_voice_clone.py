#!/usr/bin/env python3
"""Быстрый тест voice clone - только первый сегмент"""

import json
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import time

cache_dir = Path("downloads/.cache_5_Simple_Steps_for_S_f5d99c9c")
translated_file = cache_dir / "translated.json"
reference_audio = Path("D:/temp/test.wav")

# Загружаем перевод
with open(translated_file, "r", encoding="utf-8") as f:
    segments = json.load(f)

# Группируем по паузам
def merge_segments_by_pause(segments, max_pause=0.5, min_duration=10.0, max_duration=60.0):
    if not segments:
        return []
    merged = []
    current_group = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "translated": segments[0]["translated"],
    }
    for seg in segments[1:]:
        pause = seg["start"] - current_group["end"]
        current_duration = current_group["end"] - current_group["start"]
        new_duration = seg["end"] - current_group["start"]
        should_merge = (
            (pause <= max_pause and new_duration <= max_duration) or
            (current_duration < min_duration and new_duration <= max_duration)
        )
        if should_merge:
            current_group["end"] = seg["end"]
            current_group["translated"] += " " + seg["translated"]
        else:
            merged.append(current_group)
            current_group = {
                "start": seg["start"],
                "end": seg["end"],
                "translated": seg["translated"],
            }
    merged.append(current_group)
    return merged

merged = merge_segments_by_pause(segments)
print(f"Всего групп: {len(merged)}")

# Берём ГРУППУ G2
group_idx = 2
grp = merged[group_idx]
text = grp["translated"]
print(f"\nГруппа G{group_idx}:")
print(f"  Длительность: {grp['end'] - grp['start']:.1f}s")
print(f"  Символов: {len(text)}")
print(f"  Текст: {text[:200]}...")

# Загружаем модель
print("\nЗагрузка модели...")
from qwen_tts import Qwen3TTSModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map=device,
    dtype=dtype,
    attn_implementation="flash_attention_2",
)
print("Модель загружена")

# Инфо о GPU
if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Используется: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Генерируем
print("\nГенерация с voice clone...")
print(f"  Текст: {len(text)} символов")
print(f"  Ожидаемая длительность аудио: ~{grp['end'] - grp['start']:.1f}s")
print(f"  (12 Hz модель = ~12 токенов/сек аудио)")
estimated_tokens = int((grp['end'] - grp['start']) * 12)
print(f"  Ожидаемое кол-во токенов: ~{estimated_tokens}")
t0 = time.time()

wavs, sr = model.generate_voice_clone(
    text=text,
    language="Russian",
    ref_audio=str(reference_audio),
    x_vector_only_mode=True,  # Не требует транскрипт референса
    non_streaming_mode=True,  # Отключаем симуляцию стриминга
    # Параметры генерации
    max_new_tokens=4096,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
)

gen_time = time.time() - t0
audio_duration = len(wavs[0]) / sr
tokens_generated = int(audio_duration * 12)
print(f"Готово за {gen_time:.1f}s")
print(f"  Аудио: {audio_duration:.1f}s")
print(f"  Токенов: ~{tokens_generated}")
print(f"  Скорость: {tokens_generated / gen_time:.1f} токенов/сек")
print(f"  RTF (Real-Time Factor): {gen_time / audio_duration:.2f}x")

# Сохраняем
output_file = cache_dir / f"test_clone_group{group_idx}.wav"
sf.write(str(output_file), wavs[0], sr)
print(f"\nСохранено: {output_file}")
print(f"Длительность: {len(wavs[0]) / sr:.1f}s")

del model
torch.cuda.empty_cache()

# Конвертируем в mp3
import subprocess
mp3_file = cache_dir / f"test_clone_group{group_idx}.mp3"
subprocess.run(["ffmpeg", "-y", "-i", str(output_file), "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_file)], capture_output=True)
print(f"MP3: {mp3_file}")
