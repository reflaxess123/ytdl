#!/usr/bin/env python3
"""
Video Dubbing Pipeline
Транскрибирует видео, переводит на русский и озвучивает с помощью Qwen3-TTS

Pipeline:
1. Извлечение аудио из видео (FFmpeg)
2. Транскрипция (Whisper Large-V3-Turbo)
3. Перевод на русский (GLM 4.7 API)
4. Синтез речи (Qwen3-TTS)
5. Замена аудио в видео (FFmpeg)
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import hashlib
import re
from pathlib import Path
from typing import Optional

# Загружаем .env
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").strip().split('\n'):
        if '=' in _line and not _line.startswith('#'):
            _key, _value = _line.split('=', 1)
            os.environ.setdefault(_key.strip(), _value.strip())

import torch
import soundfile as sf
import whisper
from openai import OpenAI


# ============= Конфигурация =============
# GLM API (по умолчанию)
TRANSLATE_API_BASE = os.getenv("GLM_API_BASE", "https://api.z.ai/api/coding/paas/v4")
TRANSLATE_API_KEY = os.getenv("GLM_API_KEY", "")
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "glm-4.5-air")

WHISPER_MODEL = "large-v3-turbo"
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # 0.6B быстрее чем 1.7B


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Извлекает аудио из видео файла"""
    print(f"[1/5] Извлечение аудио из видео...")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"      Аудио сохранено: {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"      Ошибка FFmpeg: {e.stderr.decode()}", file=sys.stderr)
        return False


def transcribe_audio(audio_path: str, model_name: str = WHISPER_MODEL) -> dict:
    """Транскрибирует аудио с помощью Whisper"""
    print(f"[2/5] Транскрипция аудио (модель: {model_name})...")

    # Загружаем модель
    model = whisper.load_model(model_name)

    # Транскрибируем с таймстемпами
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False
    )

    print(f"      Язык определен: {result['language']}")
    print(f"      Сегментов: {len(result['segments'])}")

    # Выгружаем модель из GPU
    del model
    torch.cuda.empty_cache()
    print(f"      Whisper выгружен из GPU")

    return result


def translate_batch(
    texts: list[str],
    target_lang: str = "Russian",
    api_key: str = TRANSLATE_API_KEY,
    api_base: str = TRANSLATE_API_BASE,
    debug: bool = False
) -> list[str]:
    """Переводит батч текстов за один запрос через GLM API"""
    import re

    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    # Формируем текст с нумерацией
    numbered_text = "\n".join(f"[{i+1}] {text}" for i, text in enumerate(texts))

    system_prompt = f"""You are a professional translator. Translate each numbered line to {target_lang}.
Keep the same numbering format [N] for each line.
Keep the meaning, tone and style. Do not add explanations.
Output ONLY the translated lines with their numbers, nothing else."""

    response = client.chat.completions.create(
        model=TRANSLATE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": numbered_text}
        ],
        temperature=0.3
    )

    result_text = response.choices[0].message.content.strip()

    if debug:
        print(f"        DEBUG response:\n{result_text[:500]}...")

    # Парсим ответ с regex - ищем паттерн [число] текст
    translations = {}
    for match in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)', result_text, re.DOTALL):
        num = int(match.group(1))
        text = match.group(2).strip()
        if num <= len(texts):
            translations[num] = text

    # Собираем результат в правильном порядке
    result = []
    missing = []
    for i in range(1, len(texts) + 1):
        if i in translations:
            result.append(translations[i])
        else:
            missing.append(i)
            result.append(texts[i-1])  # fallback на оригинал

    if missing and debug:
        print(f"        MISSING translations for: {missing}")

    # Если много пропусков - пробуем альтернативный парсинг по строкам
    if len(missing) > len(texts) // 2:
        lines = [line.strip() for line in result_text.split("\n") if line.strip()]
        # Убираем номера если есть
        clean_lines = []
        for line in lines:
            if line.startswith("[") and "]" in line:
                line = line[line.index("]")+1:].strip()
            clean_lines.append(line)

        if len(clean_lines) == len(texts):
            if debug:
                print(f"        Using line-by-line fallback")
            return clean_lines

    return result


def translate_segments(
    segments: list,
    api_key: str = TRANSLATE_API_KEY,
    api_base: str = TRANSLATE_API_BASE,
    batch_size: int = 20
) -> list:
    """Переводит все сегменты транскрипции батчами"""
    print(f"[3/5] Перевод на русский язык (батчами по {batch_size})...")

    # Фильтруем пустые сегменты
    valid_segments = [(i, seg) for i, seg in enumerate(segments) if seg["text"].strip()]
    total = len(valid_segments)

    translated_segments = []

    # Разбиваем на батчи
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = valid_segments[batch_start:batch_end]

        texts = [seg["text"].strip() for _, seg in batch]

        print(f"      Перевод батча {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(texts)} сегментов)...")

        try:
            translations = translate_batch(texts, api_key=api_key, api_base=api_base)
        except Exception as e:
            print(f"      Ошибка перевода батча: {e}")
            translations = texts  # fallback на оригинал

        for (orig_idx, seg), translated in zip(batch, translations):
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["text"].strip(),
                "translated": translated
            })

    print(f"      Переведено сегментов: {len(translated_segments)}")

    # Проверяем непереведённые (без кириллицы) и ретраим
    import re
    untranslated = []
    for i, seg in enumerate(translated_segments):
        if not re.search('[а-яА-ЯёЁ]', seg["translated"]):
            untranslated.append(i)

    if untranslated:
        print(f"      Непереведённых: {len(untranslated)}, ретраим по одному...")
        for idx in untranslated:
            seg = translated_segments[idx]
            text = seg["original"]
            try:
                result = translate_batch([text], api_key=api_key, api_base=api_base)
                if result and re.search('[а-яА-ЯёЁ]', result[0]):
                    translated_segments[idx]["translated"] = result[0]
                    print(f"        [{idx}] OK")
                else:
                    print(f"        [{idx}] всё ещё английский")
            except Exception as e:
                print(f"        [{idx}] ошибка: {e}")

    return translated_segments


def merge_segments_by_pause(
    segments: list,
    max_pause: float = 0.5,
    min_duration: float = 10.0,
    max_duration: float = 60.0
) -> list:
    """
    Объединяет сегменты с маленькими паузами между ними.

    Args:
        segments: список сегментов с полями start, end, translated
        max_pause: максимальная пауза для объединения (секунды)
        min_duration: минимальная длительность группы (будет добирать сегменты)
        max_duration: максимальная длительность объединённого сегмента

    Returns:
        список объединённых сегментов
    """
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

        # Объединяем если:
        # 1. Пауза маленькая И не превышаем макс. длительность
        # 2. ИЛИ текущая группа меньше минимума (принудительно добираем)
        should_merge = (
            (pause <= max_pause and new_duration <= max_duration) or
            (current_duration < min_duration and new_duration <= max_duration)
        )

        if should_merge:
            current_group["end"] = seg["end"]
            current_group["translated"] += " " + seg["translated"]
            current_group["original_indices"].append(i)
        else:
            # Сохраняем текущую группу и начинаем новую
            merged.append(current_group)
            current_group = {
                "start": seg["start"],
                "end": seg["end"],
                "translated": seg["translated"],
                "original_indices": [i]
            }

    # Добавляем последнюю группу
    merged.append(current_group)

    return merged


def synthesize_speech(
    translated_segments: list,
    output_audio_path: str,
    original_audio_path: str,
    cache_dir: Optional[Path] = None,
    batch_size: int = 1,  # autoregressive - батчи не ускоряют
    max_pause: float = 0.5,
    max_group_duration: float = 60.0,
    reference_audio: Optional[str] = None  # голос-эталон для клонирования
) -> bool:
    """
    Синтезирует речь из переведенных сегментов с помощью Qwen3-TTS.
    Объединяет сегменты с маленькими паузами для более естественного голоса.
    Если указан reference_audio - клонирует голос из файла.
    """
    print(f"[4/5] Синтез русской речи (модель: {TTS_MODEL})...")

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        print("      Ошибка: qwen-tts не установлен.", file=sys.stderr)
        return False

    import numpy as np
    import time
    import gc
    import warnings
    warnings.filterwarnings("ignore", message=".*pad_token_id.*")

    # Объединяем сегменты по паузам
    min_group_duration = 10.0  # минимум 10 сек на группу
    merged_segments = merge_segments_by_pause(
        translated_segments,
        max_pause=max_pause,
        min_duration=min_group_duration,
        max_duration=max_group_duration
    )
    print(f"      Объединено {len(translated_segments)} -> {len(merged_segments)} групп")
    print(f"      Параметры: пауза < {max_pause}s, длительность {min_group_duration}-{max_group_duration}s")

    # Показываем статистику по группам
    group_sizes = [len(g["original_indices"]) for g in merged_segments]
    avg_size = sum(group_sizes) / len(group_sizes)
    print(f"      Сегментов в группе: мин={min(group_sizes)}, макс={max(group_sizes)}, средн={avg_size:.1f}")

    # Папка для кеша
    if cache_dir:
        segments_cache = cache_dir / "tts_merged"
        segments_cache.mkdir(exist_ok=True)
    else:
        segments_cache = None

    total = len(merged_segments)

    # Проверяем какие группы уже есть в кеше
    groups_to_generate = []
    if segments_cache:
        for i in range(total):
            seg_file = segments_cache / f"group_{i:04d}.npy"
            if not seg_file.exists():
                groups_to_generate.append(i)
        cached_count = total - len(groups_to_generate)
        if cached_count > 0:
            print(f"      Из кеша: {cached_count}/{total} групп")
    else:
        groups_to_generate = list(range(total))

    if not groups_to_generate:
        print(f"      Все группы из кеша!")
    else:
        # Загружаем модель
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"      Устройство: {device}")
        print(f"      Осталось сгенерировать: {len(groups_to_generate)} групп")

        # Выбираем модель в зависимости от режима
        use_voice_clone = reference_audio is not None
        if use_voice_clone:
            print(f"      Режим: Voice Clone (эталон: {reference_audio})")
            model_name = TTS_MODEL  # Base модель для клонирования
        else:
            print(f"      Режим: Voice Design")
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

        try:
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device,
                dtype=dtype,
                attn_implementation="flash_attention_2",
            )
            print(f"      Flash Attention 2 включен")
        except Exception as e:
            print(f"      FlashAttn недоступен: {e}")
            model = Qwen3TTSModel.from_pretrained(
                model_name, device_map=device, dtype=dtype,
            )

        # Генерируем батчами
        num_batches = (len(groups_to_generate) + batch_size - 1) // batch_size
        total_gen_time = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(groups_to_generate))
            batch_group_indices = groups_to_generate[batch_start:batch_end]
            batch_texts = [merged_segments[i]["translated"] for i in batch_group_indices]

            # Показываем инфо о батче
            total_chars = sum(len(t) for t in batch_texts)
            print(f"\n      Батч {batch_idx + 1}/{num_batches}: {len(batch_texts)} групп, {total_chars} символов")
            for i, grp_idx in enumerate(batch_group_indices):
                grp = merged_segments[grp_idx]
                txt = grp["translated"]
                n_orig = len(grp["original_indices"])
                dur = grp["end"] - grp["start"]
                print(f"        [G{grp_idx}] {n_orig} сегм, {dur:.1f}s, {len(txt)} chars: {txt[:60]}{'...' if len(txt) > 60 else ''}")

            try:
                t0 = time.time()
                print(f"      -> Генерация...")

                # Для одного элемента передаём строку, не список (быстрее)
                if len(batch_texts) == 1:
                    if use_voice_clone:
                        wavs, sr = model.generate_voice_clone(
                            text=batch_texts[0],
                            language="Russian",
                            ref_audio=reference_audio,
                            x_vector_only_mode=True,
                        )
                    else:
                        wavs, sr = model.generate_voice_design(
                            text=batch_texts[0],
                            language="Russian",
                            instruct="Говори четко и естественно, как диктор."
                        )
                else:
                    if use_voice_clone:
                        wavs, sr = model.generate_voice_clone(
                            text=batch_texts,
                            language=["Russian"] * len(batch_texts),
                            ref_audio=reference_audio,
                            x_vector_only_mode=True,
                        )
                    else:
                        wavs, sr = model.generate_voice_design(
                            text=batch_texts,
                            language=["Russian"] * len(batch_texts),
                            instruct=["Говори четко и естественно, как диктор."] * len(batch_texts)
                        )

                gen_time = time.time() - t0
                total_gen_time += gen_time
                print(f"      -> Готово за {gen_time:.1f}s ({gen_time/len(batch_texts):.1f}s/группу)")

                # Сохраняем (wavs может быть списком или одиночным массивом)
                if len(batch_texts) == 1:
                    wav_list = [wavs[0]] if isinstance(wavs, list) else [wavs]
                else:
                    wav_list = wavs

                for i, grp_idx in enumerate(batch_group_indices):
                    if segments_cache:
                        seg_file = segments_cache / f"group_{grp_idx:04d}.npy"
                        np.save(seg_file, wav_list[i])
                        # Также сохраняем WAV для прослушивания
                        wav_file = segments_cache / f"group_{grp_idx:04d}.wav"
                        import soundfile as sf
                        sf.write(str(wav_file), wav_list[i], sr)
                        print(f"      -> Сохранено: {wav_file.name}")
                del wavs, wav_list

                # Очистка памяти
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"      -> GPU память: {mem_used:.1f}GB used, {mem_reserved:.1f}GB reserved")

            except Exception as e:
                print(f"      Ошибка батча: {e}, пробуем по одному...")
                for grp_idx in batch_group_indices:
                    text = merged_segments[grp_idx]["translated"]
                    print(f"        [G{grp_idx}] ({len(text)} chars)...")
                    t0 = time.time()
                    try:
                        if use_voice_clone:
                            wavs, sr = model.generate_voice_clone(
                                text=text, language="Russian",
                                ref_audio=reference_audio,
                                x_vector_only_mode=True,
                            )
                        else:
                            wavs, sr = model.generate_voice_design(
                                text=text, language="Russian",
                                instruct="Говори четко и естественно, как диктор."
                            )
                        wav = wavs[0]
                        print(f"        -> {time.time() - t0:.1f}s")
                    except Exception as e2:
                        print(f"        -> Ошибка: {e2}")
                        wav = np.zeros(1000, dtype=np.float32)

                    if segments_cache:
                        seg_file = segments_cache / f"group_{grp_idx:04d}.npy"
                        np.save(seg_file, wav)
                        # Также сохраняем WAV для прослушивания
                        wav_file = segments_cache / f"group_{grp_idx:04d}.wav"
                        import soundfile as sf
                        sf.write(str(wav_file), wav, sr)
                        print(f"        -> Сохранено: {wav_file.name}")
                    del wav

        print(f"\n      Всего генерация: {total_gen_time:.1f}s")

        # Выгружаем модель
        del model
        torch.cuda.empty_cache()
        print(f"      TTS модель выгружена из GPU")

    # Собираем финальное аудио
    print(f"      Сборка финального аудио из {total} групп...")

    total_duration = translated_segments[-1]["end"]
    sample_rate = 24000
    total_samples = int(total_duration * sample_rate)
    final_audio = np.zeros(total_samples, dtype=np.float32)

    for i, grp in enumerate(merged_segments):
        seg_file = segments_cache / f"group_{i:04d}.npy"
        if not seg_file.exists():
            continue

        wav = np.load(seg_file)
        start_sample = int(grp["start"] * sample_rate)

        # Не выходим за границы
        if start_sample + len(wav) > total_samples:
            wav = wav[:total_samples - start_sample]

        end_sample = start_sample + len(wav)
        final_audio[start_sample:end_sample] = wav

    sf.write(output_audio_path, final_audio, sample_rate)
    print(f"      Аудио сохранено: {output_audio_path}")

    return True


def merge_audio_video(video_path: str, audio_path: str, output_path: str) -> bool:
    """Заменяет аудио дорожку в видео"""
    print(f"[5/5] Сборка финального видео...")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"      Видео сохранено: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"      Ошибка FFmpeg: {e.stderr.decode()}", file=sys.stderr)
        return False


def save_transcript(
    segments: list,
    output_path: str,
    format: str = "json"
):
    """Сохраняет транскрипцию и перевод в файл"""
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    elif format == "srt":
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{seg['translated']}\n\n")

    print(f"      Транскрипция сохранена: {output_path}")


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды в SRT формат HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def dub_video(
    video_path: str,
    output_path: Optional[str] = None,
    whisper_model: str = WHISPER_MODEL,
    keep_temp: bool = False,
    save_transcript_path: Optional[str] = None,
    api_key: str = TRANSLATE_API_KEY,
    api_base: str = TRANSLATE_API_BASE,
    no_cache: bool = False,
    reference_audio: Optional[str] = None
) -> bool:
    """
    Полный пайплайн дублирования видео

    Args:
        video_path: Путь к исходному видео
        output_path: Путь для выходного видео (по умолчанию: video_dubbed.mp4)
        whisper_model: Модель Whisper для транскрипции
        keep_temp: Сохранять временные файлы
        save_transcript_path: Путь для сохранения транскрипции (JSON)
        api_key: API ключ для перевода
        api_base: URL API для перевода
        no_cache: Игнорировать кеш и переделать всё заново
        reference_audio: Путь к аудио-эталону для клонирования голоса

    Returns:
        True если успешно, False в случае ошибки
    """
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Ошибка: файл не найден: {video_path}", file=sys.stderr)
        return False

    # Определяем пути
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_dubbed.mp4"
    else:
        output_path = Path(output_path)

    # Кеш-директория: короткое имя + хеш
    video_name = video_path.stem
    # Берём первые 20 символов + короткий хеш
    short_name = re.sub(r'[^\w\-]', '_', video_name[:20])
    name_hash = hashlib.md5(video_name.encode()).hexdigest()[:8]
    cache_dir = video_path.parent / f".cache_{short_name}_{name_hash}"
    cache_dir.mkdir(exist_ok=True)

    # Пути к кеш-файлам
    cache_audio = cache_dir / "extracted.wav"
    cache_transcript = cache_dir / "transcript.json"
    cache_translated = cache_dir / "translated.json"
    cache_dubbed = cache_dir / "dubbed.wav"

    print(f"\n{'='*60}")
    print(f"Дублирование видео: {video_path.name}")
    print(f"Кеш: {cache_dir}")
    print(f"{'='*60}\n")

    # 1. Извлекаем аудио (или берём из кеша)
    if not no_cache and cache_audio.exists():
        print(f"[1/5] Аудио из кеша: {cache_audio}")
        extracted_audio = cache_audio
    else:
        if not extract_audio(str(video_path), str(cache_audio)):
            return False
        extracted_audio = cache_audio

    # 2. Транскрибируем (или берём из кеша)
    if not no_cache and cache_transcript.exists():
        print(f"[2/5] Транскрипция из кеша: {cache_transcript}")
        with open(cache_transcript, "r", encoding="utf-8") as f:
            transcript_segments = json.load(f)
    else:
        try:
            transcript = transcribe_audio(str(extracted_audio), whisper_model)
            transcript_segments = transcript["segments"]
            # Сохраняем в кеш
            with open(cache_transcript, "w", encoding="utf-8") as f:
                json.dump(transcript_segments, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка транскрипции: {e}", file=sys.stderr)
            return False

    # 3. Переводим (или берём из кеша)
    if not no_cache and cache_translated.exists():
        print(f"[3/5] Перевод из кеша: {cache_translated}")
        with open(cache_translated, "r", encoding="utf-8") as f:
            translated_segments = json.load(f)
    else:
        try:
            translated_segments = translate_segments(
                transcript_segments,
                api_key=api_key,
                api_base=api_base
            )
            # Сохраняем в кеш
            with open(cache_translated, "w", encoding="utf-8") as f:
                json.dump(translated_segments, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка перевода: {e}", file=sys.stderr)
            return False

    # Сохраняем транскрипцию если нужно
    if save_transcript_path:
        save_transcript(translated_segments, save_transcript_path)

    # 4. Синтезируем речь (или берём из кеша)
    if not no_cache and cache_dubbed.exists():
        print(f"[4/5] Озвучка из кеша: {cache_dubbed}")
    else:
        if not synthesize_speech(translated_segments, str(cache_dubbed), str(extracted_audio), cache_dir=cache_dir, reference_audio=reference_audio):
            return False

    # 5. Собираем видео
    if not merge_audio_video(str(video_path), str(cache_dubbed), str(output_path)):
        return False

    print(f"\n{'='*60}")
    print(f"Дублирование завершено!")
    print(f"Результат: {output_path}")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Автоматическое дублирование видео на русский язык',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python dub_video.py video.mp4
  python dub_video.py video.mp4 -o video_ru.mp4
  python dub_video.py video.mp4 --whisper-model turbo --save-transcript transcript.json

Требования:
  - FFmpeg (в PATH)
  - whisper (pip install openai-whisper)
  - qwen-tts (pip install qwen-tts)
  - openai (pip install openai)
  - CUDA GPU рекомендуется для ускорения
        """
    )

    parser.add_argument(
        'video',
        help='Путь к видео файлу'
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Путь для выходного видео (по умолчанию: video_dubbed.mp4)'
    )

    parser.add_argument(
        '--whisper-model',
        default=WHISPER_MODEL,
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'large-v3-turbo', 'turbo'],
        help=f'Модель Whisper (по умолчанию: {WHISPER_MODEL})'
    )

    parser.add_argument(
        '--save-transcript',
        default=None,
        help='Сохранить транскрипцию и перевод в JSON файл'
    )

    parser.add_argument(
        '--api-key',
        default=None,
        help='API ключ (по умолчанию: Ollama не требует)'
    )

    parser.add_argument(
        '--api-base',
        default=None,
        help='URL API (по умолчанию: http://localhost:11434/v1 для Ollama)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Игнорировать кеш и переделать всё заново'
    )

    parser.add_argument(
        '--voice', '--reference-audio',
        default=None,
        dest='reference_audio',
        help='Путь к аудио-эталону для клонирования голоса (ogg/wav/mp3)'
    )

    args = parser.parse_args()

    # Используем аргументы или глобальные значения
    api_key = args.api_key or TRANSLATE_API_KEY
    api_base = args.api_base or TRANSLATE_API_BASE

    success = dub_video(
        video_path=args.video,
        output_path=args.output,
        whisper_model=args.whisper_model,
        save_transcript_path=args.save_transcript,
        api_key=api_key,
        api_base=api_base,
        no_cache=args.no_cache,
        reference_audio=args.reference_audio
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
