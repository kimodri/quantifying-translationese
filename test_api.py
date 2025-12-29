import os
import requests

api_key = os.getenv('DEEPL_API_KEY')
if not api_key:
    print('No API key')
    exit()

url = 'https://api.deepl.com/v2/translate'
headers = {'Authorization': f'DeepL-Auth-Key {api_key}'}

data = {'text': 'Hello world', 'target_lang': 'TL', 'enable_beta_languages': '1'}
response = requests.post(url, headers=headers, data=data)
print('Status:', response.status_code)
print('Response:', response.text)