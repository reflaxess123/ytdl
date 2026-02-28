# YouTube Downloader

Скачивание видео, аудио, субтитров с YouTube. Поддержка плейлистов, каналов, прокси для РФ.

## Возможности

- **Видео** — скачивание в 720p / 1080p (MP4)
- **Аудио** — извлечение в MP3 (64 kbps, минимальный размер для транскрипции)
- **Субтитры** — скачивание автоматических и ручных субтитров в `.srt`
- **Список видео** — парсинг канала/плейлиста в CSV (название + ссылка)
- **Субтитры в текст** — конвертация SRT в чистый `.txt` с разбивкой по 5-минутным блокам (для LLM)
- **Плейлисты и каналы** — пакетное скачивание с нумерацией и подпапками
- **Несколько URL** — передача нескольких ссылок через пробел
- **Прокси** — встроенная поддержка для РФ (флаг `--ru`), HTTP и SOCKS5
- **Кэш** — `.archive.txt` пропускает уже скачанные видео при повторном запуске
- **Rate limit защита** — задержка 3-6 сек между запросами

## Установка

```bash
# Через uv (рекомендуется)
uv run python download.py URL

# Или вручную
pip install yt-dlp
python download.py URL
```

## Скачивание видео

```bash
# В 1080p (по умолчанию)
python download.py URL

# В 720p
python download.py URL -q 720

# В свою папку
python download.py URL -o ./my_videos

# Через прокси для РФ
python download.py URL --ru

# Свой прокси
python download.py URL -p http://ip:port
python download.py URL -p socks5://ip:port
```

## Скачивание аудио

```bash
# MP3 одного видео
python download.py URL --mp3

# MP3 через прокси
python download.py URL --mp3 --ru

# Весь плейлист в MP3
python download.py PLAYLIST_URL --mp3 --ru
```

## Скачивание субтитров

Скачивает автоматические + ручные субтитры в `.srt` без скачивания видео.

```bash
# Субтитры одного видео (русские)
python download.py URL --subs --ru

# Субтитры на английском
python download.py URL --subs --subs-lang en --ru

# Субтитры всего плейлиста
python download.py PLAYLIST_URL --subs --ru

# Субтитры всего канала
python download.py https://www.youtube.com/@ChannelName --subs --ru
```

## Список видео канала

Парсит канал или плейлист и сохраняет CSV с названиями и ссылками.

```bash
# Список видео канала
python download.py https://www.youtube.com/@ChannelName --list --ru

# Список видео плейлиста
python download.py PLAYLIST_URL --list --ru
```

Создаёт файл `downloads/ChannelName.csv`:
```
title,url
Название видео 1,https://www.youtube.com/watch?v=...
Название видео 2,https://www.youtube.com/watch?v=...
```

## Субтитры в текст (для LLM)

Скачивает субтитры и конвертирует в чистый текст с разбивкой по временным блокам. Удобно для скармливания в LLM.

```bash
# Одно видео
python download.py URL --text --ru

# Несколько видео сразу
python download.py URL1 URL2 URL3 --text --ru

# Плейлист целиком
python download.py PLAYLIST_URL --text --ru

# Блоки по 10 минут (по умолчанию 5)
python download.py URL --text --chunk 10 --ru
```

Сохраняет в `downloads/out-text/Название.txt`:
```
[00:00]
Текст первых 5 минут сплошным текстом...

[05:00]
Текст с 5 по 10 минуту...

[10:00]
Текст с 10 по 15 минуту...
```

## Настройка прокси (для РФ)

Создайте файл `.env` в корне проекта:

```
YTDL_PROXY=http://ip:port
```

Флаг `--ru` подхватит прокси из `.env` автоматически. Также можно задать через переменную окружения `YTDL_PROXY`.

## Параметры download.py

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `url` | URL видео / плейлиста / канала | - |
| `-q, --quality` | Качество видео: 720 или 1080 | 1080 |
| `-o, --output` | Папка для сохранения | ./downloads |
| `--mp3` | Скачать только аудио в MP3 | - |
| `--subs` | Скачать только субтитры в .srt | - |
| `--subs-lang` | Язык субтитров | ru |
| `--list` | Список видео канала/плейлиста в CSV | - |
| `--text` | Субтитры -> чистый текст (out-text/*.txt) | - |
| `--chunk` | Размер блока текста в минутах для --text | 5 |
| `--ru` | Прокси для РФ (из `YTDL_PROXY` / `.env`) | - |
| `-p, --proxy` | Свой прокси-сервер | - |
| `-c, --cookies` | Куки из браузера (chrome/firefox/edge) | - |
| `extra_urls` | Дополнительные URL через пробел | - |

## Дублирование видео

Автоматически транскрибирует, переводит на русский и озвучивает видео.

```bash
python dub_video.py video.mp4
python dub_video.py video.mp4 -o video_ru.mp4
python dub_video.py video.mp4 --whisper-model turbo
python dub_video.py video.mp4 --save-transcript transcript.json
```

### Pipeline дублирования

1. **FFmpeg** - извлечение аудио из видео
2. **Whisper Large-V3-Turbo** - транскрипция (локально)
3. **GLM 4.7 API** - перевод на русский
4. **Qwen3-TTS** - синтез русской речи (локально, GPU)
5. **FFmpeg** - сборка финального видео

### Требования для дублирования

- NVIDIA GPU с CUDA (рекомендуется 8GB+ VRAM)
- ~10GB для моделей Whisper + Qwen3-TTS
- FFmpeg в PATH

### Конфигурация API

```bash
export GLM_API_KEY="your_key_here"
export GLM_API_BASE="https://api.z.ai/api/coding/paas/v4"
```

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `video` | Путь к видео файлу | - |
| `-o, --output` | Выходной файл | video_dubbed.mp4 |
| `--whisper-model` | Модель Whisper | large-v3-turbo |
| `--save-transcript` | Сохранить транскрипцию | - |
| `--api-key` | API ключ GLM | env |
| `--api-base` | URL API GLM | env |

## Лицензия

MIT. Соблюдайте авторские права при скачивании контента.
