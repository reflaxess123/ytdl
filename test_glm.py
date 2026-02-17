#!/usr/bin/env python3
"""Тест GLM API для перевода"""

import os
from pathlib import Path

# Загружаем .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().strip().split('\n'):
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

from openai import OpenAI

# GLM API
api_base = os.getenv('GLM_API_BASE', 'https://api.z.ai/api/coding/paas/v4')
api_key = os.getenv('GLM_API_KEY', '')

print(f'API Base: {api_base}')
print(f'API Key: {api_key[:10]}...' if api_key else 'API Key: НЕ ЗАДАН')

if not api_key:
    print('Задай GLM_API_KEY!')
    exit(1)

client = OpenAI(api_key=api_key, base_url=api_base)

# Тестовый батч из 5 сегментов
texts = [
    'Dynamic programming is one of the most important',
    'The main idea is to solve the problem by breaking it down',
    'into smaller subproblems that are easier to solve.',
    'Not surprisingly, this approach is very useful',
    'for solving optimization problems.'
]

numbered_text = '\n'.join(f'[{i+1}] {text}' for i, text in enumerate(texts))

system_prompt = '''You are a professional translator. Translate each numbered line to Russian.
Keep the same numbering format [N] for each line.
Keep the meaning, tone and style. Do not add explanations.
Output ONLY the translated lines with their numbers, nothing else.'''

print('\n=== ЗАПРОС ===')
print(numbered_text)
print('\n=== ОТПРАВКА... ===')

response = client.chat.completions.create(
    model='glm-4.5-air',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': numbered_text}
    ],
    temperature=0.3
)

result = response.choices[0].message.content.strip()
print('\n=== ОТВЕТ ===')
print(result)
print('\n=== ПАРСИНГ ===')
for line in result.split('\n'):
    line = line.strip()
    if line:
        print(f'  LINE: {repr(line)}')
