#!/usr/bin/env python3
"""
YouTube Video Downloader
Скачивает видео с YouTube в качестве 720p или 1080p
"""

import os
import sys
import argparse
from pathlib import Path
import yt_dlp

# Загружаем .env если есть
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

DEFAULT_PROXY = os.environ.get('YTDL_PROXY', '')


def list_videos(url: str, output_dir: str = "./downloads", proxy: str = None):
    """Парсит канал/плейлист и сохраняет список видео в CSV"""
    import csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'ignoreerrors': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    if proxy:
        opts['proxy'] = proxy

    print(f"📋 Получаю список видео...")
    print(f"🔗 URL: {url}")
    if proxy:
        print(f"🌐 Прокси: {proxy}")

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get('title', 'videos')
    entries = list(info.get('entries', []))

    # Для каналов title может содержать " - Videos", убираем
    clean_title = title.replace(' - Videos', '').replace(' - Видео', '').strip()
    safe_title = "".join(c if c.isalnum() or c in ' .-_' else '_' for c in clean_title)
    csv_path = output_path / f"{safe_title}.csv"

    rows = []
    for entry in entries:
        vid_title = entry.get('title', '')
        vid_id = entry.get('id', '')
        vid_url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id else entry.get('url', '')
        if vid_title or vid_id:
            rows.append((vid_title, vid_url))

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'url'])
        writer.writerows(rows)

    print(f"\n✅ Сохранено {len(rows)} видео в {csv_path}")


def srt_to_text(srt_path: Path, chunk_minutes: int = 5) -> str:
    """Конвертирует SRT в чистый текст с разбивкой по интервалам"""
    import re

    content = srt_path.read_text(encoding='utf-8', errors='replace')

    entries = []
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        # Парсим таймкод: 00:01:23,456 --> 00:01:25,789
        tc_match = re.match(r'(\d{2}):(\d{2}):(\d{2})', lines[1])
        if not tc_match:
            continue
        h, m, s = int(tc_match.group(1)), int(tc_match.group(2)), int(tc_match.group(3))
        seconds = h * 3600 + m * 60 + s
        text = ' '.join(lines[2:]).strip()
        # Убираем HTML-теги и дубликаты от YouTube auto-subs
        text = re.sub(r'<[^>]+>', '', text)
        if text:
            entries.append((seconds, text))

    if not entries:
        return ''

    # Дедупликация (YouTube авто-субтитры часто дублируют строки)
    seen = set()
    unique = []
    for sec, text in entries:
        if text not in seen:
            seen.add(text)
            unique.append((sec, text))

    # Разбиваем по chunk_minutes-минутным интервалам
    chunk_sec = chunk_minutes * 60
    chunks = []
    current_chunk = []
    current_boundary = chunk_sec

    for sec, text in unique:
        if sec >= current_boundary and current_chunk:
            chunks.append((current_boundary - chunk_sec, current_chunk))
            current_chunk = []
            current_boundary = (sec // chunk_sec + 1) * chunk_sec
        current_chunk.append(text)

    if current_chunk:
        chunks.append((current_boundary - chunk_sec, current_chunk))

    # Форматируем
    parts = []
    for start_sec, texts in chunks:
        h, m = start_sec // 3600, (start_sec % 3600) // 60
        label = f"[{h:02d}:{m:02d}]" if h > 0 else f"[{m:02d}:00]"
        parts.append(f"{label}\n{' '.join(texts)}")

    return '\n\n'.join(parts)


def summarize_with_llm(txt_path: Path, output_dir: Path):
    """Отправляет текстовый файл в LLM и сохраняет развёрнутые таймкоды"""
    import json
    import urllib.request
    import urllib.error

    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if not api_key:
        print("  ⚠️  OPENROUTER_API_KEY не задан в .env")
        return

    text = txt_path.read_text(encoding='utf-8')
    if not text.strip():
        return

    summary_dir = output_dir / 'summaries'
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_file = summary_dir / txt_path.name

    if out_file.exists():
        print(f"  ⏭️  Уже есть: {out_file.name}")
        return

    prompt = f"""Ты — эксперт по анализу образовательных лекций. Тебе дана транскрипция лекции с YouTube, разбитая по временным блокам.

Твоя задача — создать МАКСИМАЛЬНО ДЕТАЛЬНОЕ содержание лекции. Для КАЖДОГО смыслового блока (их должно быть много, не объединяй темы):

1. **Таймкод** [ЧЧ:ММ] — точное время начала блока
2. **Название темы** — чёткое и конкретное название раздела
3. **Подробное описание** (5-10 предложений):
   - Какие именно понятия/теоремы/формулы вводятся
   - Какие примеры приводит лектор
   - Какие выводы делаются
   - Связь с предыдущими и последующими темами
4. **Ключевые термины** — список основных терминов и определений из блока

В конце добавь:
- **Общее резюме лекции** (10-15 предложений)
- **Полный список ключевых понятий и определений** введённых в лекции
- **Рекомендации** — что нужно знать/повторить перед следующей лекцией

Пиши на русском языке. Будь максимально точен в описании математических терминов, формул и определений. Не пропускай важные детали.

Транскрипция:
{text}"""

    body = json.dumps({
        'model': 'openai/gpt-4o-mini',
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode()

    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        content = result['choices'][0]['message']['content']
        out_file.write_text(content, encoding='utf-8')
        print(f"  🤖 {out_file.name}")
    except urllib.error.HTTPError as e:
        err = e.read().decode() if e.fp else str(e)
        print(f"  ❌ LLM ошибка ({e.code}): {err[:200]}")
    except Exception as e:
        print(f"  ❌ LLM ошибка: {e}")


def summarize_with_gemini(txt_path: Path, output_dir: Path, proxy: str = None):
    """Отправляет текстовый файл в Gemini 3.1 Pro и сохраняет развёрнутые таймкоды"""
    import json
    import urllib.request
    import urllib.error

    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY не задан в .env")
        return

    text = txt_path.read_text(encoding='utf-8')
    if not text.strip():
        return

    summary_dir = output_dir / 'summaries'
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_file = summary_dir / txt_path.name

    if out_file.exists():
        print(f"  ⏭️  Уже есть: {out_file.name}")
        return

    prompt = f"""Ты — эксперт по анализу образовательных лекций. Тебе дана транскрипция лекции с YouTube, разбитая по временным блокам.

Твоя задача — создать МАКСИМАЛЬНО ДЕТАЛЬНОЕ содержание лекции. Для КАЖДОГО смыслового блока (их должно быть много, не объединяй темы):

1. **Таймкод** [ЧЧ:ММ] — точное время начала блока
2. **Название темы** — чёткое и конкретное название раздела
3. **Подробное описание** (5-10 предложений):
   - Какие именно понятия/теоремы/формулы вводятся
   - Какие примеры приводит лектор
   - Какие выводы делаются
   - Связь с предыдущими и последующими темами
4. **Ключевые термины** — список основных терминов и определений из блока

В конце добавь:
- **Общее резюме лекции** (10-15 предложений)
- **Полный список ключевых понятий и определений** введённых в лекции
- **Рекомендации** — что нужно знать/повторить перед следующей лекцией

Пиши на русском языке. Будь максимально точен в описании математических терминов, формул и определений. Не пропускай важные детали.

Транскрипция:
{text}"""

    body = json.dumps({
        'contents': [{'parts': [{'text': prompt}]}],
    }).encode()

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={api_key}'
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
    )

    try:
        if proxy:
            proxy_handler = urllib.request.ProxyHandler({'https': proxy, 'http': proxy})
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()
        with opener.open(req, timeout=180) as resp:
            result = json.loads(resp.read())
        content = result['candidates'][0]['content']['parts'][0]['text']
        out_file.write_text(content, encoding='utf-8')
        print(f"  🤖 {out_file.name}")
    except urllib.error.HTTPError as e:
        err = e.read().decode() if e.fp else str(e)
        print(f"  ❌ Gemini ошибка ({e.code}): {err[:200]}")
    except Exception as e:
        print(f"  ❌ Gemini ошибка: {e}")


def _safe_dirname(name: str) -> str:
    """Sanitize string for use as directory name"""
    return "".join(c if c.isalnum() or c in ' .-_' else '_' for c in name).strip()


def download_descriptions(url: str, output_dir: str = "./downloads", proxy: str = None):
    """Сохраняет описания видео сразу в out-desc/*.txt"""
    output_path = Path(output_dir)

    meta_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'ignoreerrors': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    if proxy:
        meta_opts['proxy'] = proxy

    with yt_dlp.YoutubeDL(meta_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    playlist_title = info.get('title', '') if info else ''
    playlist_title = playlist_title.replace(' - Videos', '').replace(' - Видео', '').strip()

    if playlist_title:
        base = output_path / _safe_dirname(playlist_title)
    else:
        base = output_path

    desc_path = base / 'out-desc'
    desc_path.mkdir(parents=True, exist_ok=True)

    is_playlist = info.get('_type') == 'playlist'
    entries = list(info.get('entries') or [info])

    print(f"\n📝 Скачиваю описания видео...")
    if playlist_title:
        print(f"📋 Плейлист: {playlist_title}")
    print(f"🔗 URL: {url}")
    if proxy:
        print(f"🌐 Прокси: {proxy}")
    print(f"📁 Папка: {base.absolute()}\n")

    fetch_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'ignoreerrors': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    if proxy:
        fetch_opts['proxy'] = proxy

    saved = 0
    skipped = 0
    with yt_dlp.YoutubeDL(fetch_opts) as ydl:
        for i, entry in enumerate(entries, 1):
            if not entry:
                continue
            video_id = entry.get('id')
            if not video_id:
                continue

            title = entry.get('title') or f'video_{video_id}'
            safe = _safe_dirname(title)
            prefix = f'{i}. ' if is_playlist else ''
            out_file = desc_path / f'{prefix}{safe}.txt'

            if out_file.exists():
                skipped += 1
                continue

            try:
                full = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
                desc = (full.get('description') or '').strip() if full else ''
                if desc:
                    out_file.write_text(desc, encoding='utf-8')
                    saved += 1
                    print(f"  ✅ {out_file.name}")
                else:
                    print(f"  ⚠️  Пустое описание: {out_file.name}")
            except Exception as e:
                print(f"  ❌ {out_file.name}: {e}")

    print(f"\n✅ Описаний сохранено: {saved}, пропущено (уже есть): {skipped} в {desc_path}")


def download_text(url: str, output_dir: str = "./downloads", proxy: str = None, subs_lang: str = "ru", chunk_minutes: int = 5, summarize: bool = False, use_gemini: bool = False):
    """Скачивает субтитры и конвертирует в чистый текст"""
    output_path = Path(output_dir)

    # Get playlist/video title for per-playlist subdirectory
    meta_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'ignoreerrors': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    if proxy:
        meta_opts['proxy'] = proxy

    with yt_dlp.YoutubeDL(meta_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    playlist_title = info.get('title', '') if info else ''
    playlist_title = playlist_title.replace(' - Videos', '').replace(' - Видео', '').strip()

    if playlist_title:
        base = output_path / _safe_dirname(playlist_title)
    else:
        base = output_path

    srt_cache = base / 'srt-cache'
    text_path = base / 'out-text'
    srt_cache.mkdir(parents=True, exist_ok=True)
    text_path.mkdir(parents=True, exist_ok=True)

    # Download SRT to persistent cache
    ydl_opts = {
        'outtmpl': str(srt_cache / '%(playlist_index|)s%(playlist_index&. |)s%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'ignoreerrors': True,
        'sleep_interval': 3,
        'max_sleep_interval': 6,
        'download_archive': str(srt_cache / '.archive.txt'),
        'skip_download': True,
        'writeautomaticsub': True,
        'writesubtitles': True,
        'subtitleslangs': [subs_lang],
        'subtitlesformat': 'srt',
        'postprocessors': [{'key': 'FFmpegSubtitlesConvertor', 'format': 'srt'}],
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    if proxy:
        ydl_opts['proxy'] = proxy

    print(f"\n📝 Скачиваю субтитры и конвертирую в текст...")
    if playlist_title:
        print(f"📋 Плейлист: {playlist_title}")
    print(f"🔗 URL: {url}")
    if proxy:
        print(f"🌐 Прокси: {proxy}")
    print(f"📁 Папка: {base.absolute()}\n")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Convert SRT -> text (skip existing .txt)
    txt_files = []
    srt_files = sorted(srt_cache.glob(f'*.{subs_lang}.srt'))
    if not srt_files:
        print("⚠️  Субтитры не найдены")
        return

    for srt_file in srt_files:
        name = srt_file.stem
        if name.endswith(f'.{subs_lang}'):
            name = name[:-len(f'.{subs_lang}')]
        txt_file = text_path / f"{name}.txt"

        if txt_file.exists():
            print(f"  ⏭️  Уже есть: {txt_file.name}")
            txt_files.append(txt_file)
            continue

        text = srt_to_text(srt_file, chunk_minutes)
        if text:
            txt_file.write_text(text, encoding='utf-8')
            txt_files.append(txt_file)
            print(f"  ✅ {txt_file.name}")
        else:
            print(f"  ⚠️  Пустые субтитры: {srt_file.name}")

    print(f"\n✅ Тексты сохранены в {text_path}")

    if summarize and txt_files:
        import concurrent.futures
        summary_dir = base / 'summaries'
        summary_dir.mkdir(parents=True, exist_ok=True)
        summarize_fn = summarize_with_gemini if use_gemini else summarize_with_llm
        provider = "Gemini 3.1 Pro" if use_gemini else "OpenRouter GPT-4o-mini"
        print(f"\n🤖 Отправляю в {provider} для анализа ({len(txt_files)} файлов, 20 воркеров)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(summarize_fn, tf, base, *([proxy] if use_gemini else [])): tf for tf in txt_files}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  ❌ Ошибка воркера: {e}")

        # Concatenate all summaries into one file
        all_summaries = sorted(summary_dir.glob('*.txt'))
        if all_summaries:
            merged_name = f"{_safe_dirname(playlist_title)}_ALL.txt" if playlist_title else "ALL_SUMMARIES.txt"
            merged_path = base / merged_name
            parts = []
            for sf in all_summaries:
                parts.append(f"{'='*60}\n{sf.stem}\n{'='*60}\n\n{sf.read_text(encoding='utf-8')}")
            merged_path.write_text('\n\n\n'.join(parts), encoding='utf-8')
            print(f"\n📄 Все саммари склеены в: {merged_path.name}")

        print(f"✅ Саммари сохранены в {summary_dir}")


def download_video(url: str, quality: str = "1080", output_dir: str = "./downloads", mp3: bool = False, proxy: str = None, cookies_browser: str = None, subs: bool = False, subs_lang: str = "ru"):
    """
    Скачивает видео с YouTube в указанном качестве

    Args:
        url: URL видео на YouTube
        quality: Качество видео (720 или 1080)
        output_dir: Директория для сохранения видео
        mp3: Скачать только аудио в формате MP3
        proxy: Прокси-сервер (например, http://ip:port)
    """
    # Создаем директорию для загрузок, если её нет
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Базовые настройки для yt-dlp
    ydl_opts = {
        'outtmpl': str(output_path / '%(playlist_title|)s%(playlist_title&/|)s%(playlist_index|)s%(playlist_index&. |)s%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'ignoreerrors': True,
        'download_archive': str(output_path / '.archive.txt'),
        'sleep_interval': 3,
        'max_sleep_interval': 6,
        'progress_hooks': [progress_hook],
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
    }

    if proxy:
        ydl_opts['proxy'] = proxy

    if cookies_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_browser,)

    if subs:
        ydl_opts['skip_download'] = True
        ydl_opts['writeautomaticsub'] = True
        ydl_opts['writesubtitles'] = True
        ydl_opts['subtitleslangs'] = [subs_lang]
        ydl_opts['subtitlesformat'] = 'srt'
        ydl_opts['postprocessors'] = [{'key': 'FFmpegSubtitlesConvertor', 'format': 'srt'}]
    elif mp3:
        ydl_opts['format'] = 'worstaudio/worst'
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }]
    else:
        ydl_opts['format'] = f'bestvideo[height<={quality}]+bestaudio/best[height<={quality}]'
        ydl_opts['merge_output_format'] = 'mp4'

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if subs:
                print(f"📝 Скачиваю субтитры ({subs_lang})...")
            elif mp3:
                print(f"🎵 Скачиваю аудио в MP3...")
            else:
                print(f"📥 Скачиваю видео в качестве {quality}p...")
            print(f"🔗 URL: {url}")
            if proxy:
                print(f"🌐 Прокси: {proxy}")
            print(f"📁 Сохранение в: {output_path.absolute()}\n")

            # Получаем информацию о видео/плейлисте
            info = ydl.extract_info(url, download=False)

            if info.get('_type') == 'playlist':
                entries = info.get('entries', [])
                count = info.get('playlist_count') or len(list(entries))
                print(f"📋 Плейлист: {info.get('title', 'Unknown')}")
                print(f"📊 Видео в плейлисте: {count}\n")
            else:
                print(f"📹 Название: {info.get('title', 'Unknown')}")
                duration = info.get('duration', 0) or 0
                print(f"⏱️  Длительность: {duration // 60} мин {duration % 60} сек\n")

            # Скачиваем
            ydl.download([url])

            if subs:
                print("\n✅ Субтитры скачаны!")
            elif mp3:
                print("\n✅ MP3 успешно скачан!")
            else:
                print("\n✅ Видео успешно скачано!")

    except Exception as e:
        print(f"\n❌ Ошибка при скачивании: {e}", file=sys.stderr)
        sys.exit(1)


def progress_hook(d):
    """Хук для отображения прогресса скачивания"""
    if d['status'] == 'downloading':
        if d.get('total_bytes'):
            percent = d['downloaded_bytes'] / d['total_bytes'] * 100
            print(f"\r⬇️  Прогресс: {percent:.1f}%", end='', flush=True)
        elif '_percent_str' in d:
            print(f"\r⬇️  Прогресс: {d['_percent_str']}", end='', flush=True)
    elif d['status'] == 'finished':
        print(f"\n🔄 Обработка видео...")


def main():
    parser = argparse.ArgumentParser(
        description='Скачивание видео/аудио/субтитров с YouTube (видео, плейлисты, каналы)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  uv run python download.py URL                              # видео 1080p
  uv run python download.py URL -q 720                       # видео 720p
  uv run python download.py URL --mp3 --ru                   # MP3 через прокси (РФ)
  uv run python download.py URL -p http://ip:port            # свой прокси
  uv run python download.py URL -q 1080 -o ./my_videos       # в свою папку
  uv run python download.py PLAYLIST_URL --mp3 --ru          # плейлист в MP3
  uv run python download.py URL --subs --ru                  # субтитры (ru)
  uv run python download.py URL --subs --subs-lang en --ru   # субтитры (en)
  uv run python download.py @CHANNEL --subs --ru             # субтитры всего канала
  uv run python download.py @CHANNEL --list --ru             # список видео в CSV
  uv run python download.py URL --text --ru                  # субтитры -> текст
  uv run python download.py URL --desc --ru                  # описания -> текст
  uv run python download.py URL1 URL2 --text --ru            # несколько URL сразу
  uv run python download.py URL --text --summary --ru        # текст + анализ LLM

URL может быть видео, плейлистом или каналом:
  https://www.youtube.com/watch?v=VIDEO_ID
  https://www.youtube.com/playlist?list=PLAYLIST_ID
  https://www.youtube.com/@ChannelName
        """
    )

    parser.add_argument(
        'url',
        help='URL видео или плейлиста на YouTube'
    )

    parser.add_argument(
        '-q', '--quality',
        choices=['720', '1080'],
        default='1080',
        help='Качество видео (по умолчанию: 1080)'
    )

    parser.add_argument(
        '-o', '--output',
        default='./downloads',
        help='Директория для сохранения (по умолчанию: ./downloads)'
    )

    parser.add_argument(
        '--mp3',
        action='store_true',
        help='Скачать только аудио в формате MP3'
    )

    parser.add_argument(
        '-p', '--proxy',
        help='Прокси-сервер (например, http://ip:port или socks5://ip:port)'
    )

    parser.add_argument(
        '--ru',
        action='store_true',
        help='Использовать прокси из YTDL_PROXY (для РФ)'
    )

    parser.add_argument(
        '-c', '--cookies',
        choices=['chrome', 'firefox', 'edge', 'brave', 'opera'],
        help='Взять куки из браузера (chrome, firefox, edge, brave, opera)'
    )

    parser.add_argument(
        '--subs',
        action='store_true',
        help='Скачать только субтитры (авто + ручные) в .srt'
    )

    parser.add_argument(
        '--subs-lang',
        default='ru',
        help='Язык субтитров (по умолчанию: ru)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='Сохранить список видео канала/плейлиста в CSV (название, ссылка)'
    )

    parser.add_argument(
        '--text',
        action='store_true',
        help='Скачать субтитры и конвертировать в чистый текст (out-text/*.txt)'
    )

    parser.add_argument(
        '--desc',
        action='store_true',
        help='Скачать описания видео в текст (out-desc/*.txt)'
    )

    parser.add_argument(
        '--chunk',
        type=int,
        default=5,
        help='Размер блока текста в минутах для --text (по умолчанию: 5)'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Отправить текст в LLM для создания таймкодов и описания лекции'
    )

    parser.add_argument(
        '--gemini',
        action='store_true',
        help='Использовать Gemini 3.1 Pro вместо OpenRouter для --summary'
    )

    parser.add_argument(
        'extra_urls',
        nargs='*',
        help='Дополнительные URL для пакетной обработки'
    )

    args = parser.parse_args()

    proxy = args.proxy
    if args.ru:
        proxy = DEFAULT_PROXY

    urls = [args.url] + (args.extra_urls or [])

    if args.list:
        for u in urls:
            list_videos(u, args.output, proxy)
    elif args.text:
        for u in urls:
            download_text(u, args.output, proxy, args.subs_lang, args.chunk, args.summary, args.gemini)
    elif args.desc:
        for u in urls:
            download_descriptions(u, args.output, proxy)
    else:
        for u in urls:
            download_video(u, args.quality, args.output, args.mp3, proxy, args.cookies, args.subs, args.subs_lang)


if __name__ == "__main__":
    main()
