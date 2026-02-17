#!/usr/bin/env python3
"""Собирает превью аудио из уже сгенерированных сегментов"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path

cache_dir = Path("downloads/.cache_5_Simple_Steps_for_S_f5d99c9c")
segments_cache = cache_dir / "tts_segments"
translated_file = cache_dir / "translated.json"

# Загружаем перевод для таймингов
with open(translated_file, "r", encoding="utf-8") as f:
    translated_segments = json.load(f)

print(f"Всего сегментов: {len(translated_segments)}")

# Смотрим сколько уже сгенерировано
existing = list(segments_cache.glob("seg_*.npy"))
print(f"Сгенерировано: {len(existing)}")

if not existing:
    print("Нет сегментов для сборки!")
    exit(1)

# Определяем длительность по последнему сегменту
total_duration = translated_segments[-1]["end"]
sample_rate = 24000
total_samples = int(total_duration * sample_rate)

print(f"Длительность: {total_duration:.1f}s")

# Создаем буфер
final_audio = np.zeros(total_samples, dtype=np.float32)

# Вставляем сегменты
inserted = 0
for i, seg in enumerate(translated_segments):
    seg_file = segments_cache / f"seg_{i:04d}.npy"
    if not seg_file.exists():
        continue

    wav = np.load(seg_file)
    start_sample = int(seg["start"] * sample_rate)

    # Не выходим за границы
    if start_sample + len(wav) > total_samples:
        wav = wav[:total_samples - start_sample]

    end_sample = start_sample + len(wav)
    final_audio[start_sample:end_sample] = wav
    inserted += 1

print(f"Вставлено: {inserted} сегментов")

# Сохраняем
output_file = cache_dir / "preview.wav"
sf.write(str(output_file), final_audio, sample_rate)
print(f"Сохранено: {output_file}")
