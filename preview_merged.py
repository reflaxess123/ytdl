#!/usr/bin/env python3
"""Превью из объединённых групп"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path

cache_dir = Path("downloads/.cache_5_Simple_Steps_for_S_f5d99c9c")
segments_cache = cache_dir / "tts_merged"
translated_file = cache_dir / "translated.json"

# Загружаем перевод
with open(translated_file, "r", encoding="utf-8") as f:
    segments = json.load(f)

# Пересоздаём группы (та же логика)
def merge_segments_by_pause(segments, max_pause=0.5, min_duration=10.0, max_duration=60.0):
    if not segments:
        return []
    merged = []
    current_group = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "translated": segments[0]["translated"],
    }
    for i, seg in enumerate(segments[1:], 1):
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
print(f"Групп: {len(merged)}")

# Смотрим сколько готово
existing = list(segments_cache.glob("group_*.npy"))
print(f"Сгенерировано: {len(existing)}")

if not existing:
    print("Нет групп!")
    exit(1)

# Находим последнюю готовую группу
last_group_idx = -1
for i in range(len(merged)):
    if (segments_cache / f"group_{i:04d}.npy").exists():
        last_group_idx = i

if last_group_idx < 0:
    print("Нет групп!")
    exit(1)

# Длительность только до конца последней готовой группы
total_duration = merged[last_group_idx]["end"]
sample_rate = 24000
total_samples = int(total_duration * sample_rate)
final_audio = np.zeros(total_samples, dtype=np.float32)
print(f"Длительность превью: {total_duration:.1f}s")

inserted = 0
for i, grp in enumerate(merged):
    seg_file = segments_cache / f"group_{i:04d}.npy"
    if not seg_file.exists():
        continue

    wav = np.load(seg_file)
    start_sample = int(grp["start"] * sample_rate)

    if start_sample + len(wav) > total_samples:
        wav = wav[:total_samples - start_sample]

    end_sample = start_sample + len(wav)
    final_audio[start_sample:end_sample] = wav
    inserted += 1
    print(f"  G{i}: {grp['start']:.1f}s - {grp['end']:.1f}s ({len(wav)/sample_rate:.1f}s audio)")

print(f"\nВставлено: {inserted} групп")

output_file = cache_dir / "preview_merged.wav"
sf.write(str(output_file), final_audio, sample_rate)
print(f"Сохранено: {output_file}")
