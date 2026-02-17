# YouTube Video Downloader & Dubber

Скачивание видео с YouTube + автоматическое дублирование на русский язык.

## Установка

### Conda (рекомендуется для дублирования)

```bash
# Создаём окружение с PyTorch CUDA
conda env create -f environment.yml

# Активируем
conda activate dl-youtube

# Опционально: FlashAttention для ускорения TTS
pip install flash-attn --no-build-isolation
```

### Только скачивание (без дублирования)

```bash
pip install yt-dlp
```

## Скачивание видео

```bash
# В 1080p (по умолчанию)
python download.py URL

# В 720p
python download.py URL -q 720

# В свою папку
python download.py URL -o ./my_videos

# Только аудио в MP3
python download.py URL --mp3

# Через прокси для РФ (YouTube заблокирован, см. настройку ниже)
python download.py URL --ru
python download.py URL --mp3 --ru

# Свой прокси
python download.py URL -p http://ip:port
python download.py URL -p socks5://ip:port
```

## Скачивание субтитров

Скачивает субтитры (автоматические + ручные) в формате `.srt` без скачивания видео.

```bash
# Субтитры одного видео (русские)
python download.py URL --subs --ru

# Субтитры на английском
python download.py URL --subs --subs-lang en --ru

# Субтитры всего плейлиста
python download.py PLAYLIST_URL --subs --ru

# Субтитры всего канала
python download.py https://www.youtube.com/@ChannelName --subs --ru

# В свою папку
python download.py https://www.youtube.com/@ChannelName --subs --ru -o ./subs
```

## Настройка прокси (для РФ)

Создайте файл `.env` в корне проекта:

```
YTDL_PROXY=http://ip:port
```

Флаг `--ru` подхватит прокси из `.env` автоматически. Также можно задать через переменную окружения `YTDL_PROXY`.

## Дублирование видео

Автоматически транскрибирует, переводит на русский и озвучивает видео.

```bash
# Базовое использование
python dub_video.py video.mp4

# С указанием выходного файла
python dub_video.py video.mp4 -o video_ru.mp4

# Быстрая модель Whisper (менее точная)
python dub_video.py video.mp4 --whisper-model turbo

# Сохранить транскрипцию с переводом
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

API ключ можно передать через:
- Аргумент: `--api-key YOUR_KEY`
- Переменную окружения: `GLM_API_KEY`

```bash
export GLM_API_KEY="your_key_here"
export GLM_API_BASE="https://api.z.ai/api/coding/paas/v4"
```

## Параметры

### download.py
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `url` | URL видео / плейлиста / канала | - |
| `-q, --quality` | Качество: 720 или 1080 | 1080 |
| `-o, --output` | Папка для сохранения | ./downloads |
| `--mp3` | Скачать только аудио в MP3 | - |
| `--ru` | Прокси для РФ (из `YTDL_PROXY` / `.env`) | - |
| `-p, --proxy` | Свой прокси-сервер | - |
| `-c, --cookies` | Куки из браузера (chrome/firefox/edge) | - |
| `--subs` | Скачать только субтитры в .srt | - |
| `--subs-lang` | Язык субтитров | ru |

### dub_video.py
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
