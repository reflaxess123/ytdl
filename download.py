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

    args = parser.parse_args()

    proxy = args.proxy
    if args.ru:
        proxy = DEFAULT_PROXY

    if args.list:
        list_videos(args.url, args.output, proxy)
    else:
        download_video(args.url, args.quality, args.output, args.mp3, proxy, args.cookies, args.subs, args.subs_lang)


if __name__ == "__main__":
    main()
