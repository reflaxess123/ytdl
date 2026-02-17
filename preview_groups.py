#!/usr/bin/env python3
"""Превью группировки сегментов"""

import json
from pathlib import Path

cache_dir = Path("downloads/.cache_5_Simple_Steps_for_S_f5d99c9c")
translated_file = cache_dir / "translated.json"

with open(translated_file, "r", encoding="utf-8") as f:
    segments = json.load(f)

print(f"Всего сегментов: {len(segments)}")
print(f"Общая длительность: {segments[-1]['end']:.1f}s")
print()

def merge_segments_by_pause(segments, max_pause=0.5, min_duration=10.0, max_duration=60.0):
    if not segments:
        return []

    merged = []
    current_group = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "translated": segments[0]["translated"],
        "original_indices": [0]
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
            current_group["original_indices"].append(i)
        else:
            merged.append(current_group)
            current_group = {
                "start": seg["start"],
                "end": seg["end"],
                "translated": seg["translated"],
                "original_indices": [i]
            }

    merged.append(current_group)
    return merged

# Группируем
merged = merge_segments_by_pause(segments, max_pause=0.5, min_duration=10.0, max_duration=60.0)

print(f"Групп после объединения: {len(merged)}")
print(f"Параметры: пауза < 0.5s, длительность 10-60s")
print()

# Статистика
durations = [g["end"] - g["start"] for g in merged]
sizes = [len(g["original_indices"]) for g in merged]

print("=== Статистика ===")
print(f"Длительность групп:")
print(f"  мин: {min(durations):.1f}s")
print(f"  макс: {max(durations):.1f}s")
print(f"  средн: {sum(durations)/len(durations):.1f}s")
print()
print(f"Сегментов в группе:")
print(f"  мин: {min(sizes)}")
print(f"  макс: {max(sizes)}")
print(f"  средн: {sum(sizes)/len(sizes):.1f}")
print()

# Распределение по длительности
print("=== Распределение по длительности ===")
buckets = {"<10s": 0, "10-20s": 0, "20-30s": 0, "30-40s": 0, "40-50s": 0, "50-60s": 0, ">60s": 0}
for d in durations:
    if d < 10: buckets["<10s"] += 1
    elif d < 20: buckets["10-20s"] += 1
    elif d < 30: buckets["20-30s"] += 1
    elif d < 40: buckets["30-40s"] += 1
    elif d < 50: buckets["40-50s"] += 1
    elif d <= 60: buckets["50-60s"] += 1
    else: buckets[">60s"] += 1

for k, v in buckets.items():
    bar = "█" * v
    print(f"  {k:8} {v:3} {bar}")
print()

# Показываем первые 10 групп
print("=== Первые 10 групп ===")
for i, g in enumerate(merged[:10]):
    dur = g["end"] - g["start"]
    n = len(g["original_indices"])
    txt = g["translated"][:80] + "..." if len(g["translated"]) > 80 else g["translated"]
    print(f"[G{i:02d}] {dur:5.1f}s, {n:2} сегм: {txt}")
