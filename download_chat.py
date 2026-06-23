import requests
import json

# Твой ID диалога
CHAT_ID = 'T6259114687'

# URL API TMSG Support
URL = f'https://tmsg-support.tinkoff.ru/api/messages/{CHAT_ID}'

# Заголовки (нужно взять из браузера)
HEADERS = {
    'Authorization': 'Bearer YOUR_TOKEN',  # Твой токен из cookies
    'Accept': 'application/json'
}

# Запрос к API
response = requests.get(URL, headers=HEADERS)

if response.status_code == 200:
    messages = response.json()
    
    # Сохраняем в текстовый файл
    with open(f'chat_{CHAT_ID}.txt', 'w', encoding='utf-8') as f:
        for msg in messages:
            author = msg.get('author', {}).get('role', 'UNKNOWN')
            text = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            f.write(f'{timestamp} | {author} | {text}\n')
    
    print(f'Диалог сохранён в chat_{CHAT_ID}.txt')
else:
    print(f'Ошибка: {response.status_code}')
    print(response.text)