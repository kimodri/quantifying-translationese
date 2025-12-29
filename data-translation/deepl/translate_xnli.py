#!/usr/bin/env python3
"""Translate XNLI cleaned CSV using DeepL API.

Usage example:
  python data-translation/deepl/translate_xnli.py \
    --input datasets/cleaned/cleaned_xnli.csv \
    --output datasets/translated/deepl/deepl_translated_xnli.csv \
    --batch-size 10 --target-lang TL

Set your DeepL API key in the environment variable `DEEPL_API_KEY`.
"""
import argparse
import os
import pandas as pd
import requests
from tqdm import tqdm

try:
    import deepl
    import requests
except Exception:
    raise SystemExit("Install the 'deepl' and 'requests' packages (pip install deepl requests) to use this script.")

DEEPL_API_URL = "https://api.deepl.com/v2/translate"
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

if not DEEPL_API_KEY:
    raise RuntimeError("DEEPL_API_KEY environment variable not set")

def translate_text(text, target_lang="TL"):
    print("Using DeepL endpoint:", DEEPL_API_URL)
    payload = {
        "text": text,
        "target_lang": target_lang,
        "enable_beta_languages": 1
    }

    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"
    }

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.post(
                DEEPL_API_URL,
                data=payload,
                headers=headers,
                timeout=60  # Increased timeout
            )

            if response.status_code == 200:
                return response.json()["translations"][0]["text"]
            else:
                print("❌ Error:", response.status_code)
                print(response.text)
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                import time
                time.sleep(5)  # Wait 5 seconds before retry
            else:
                return None

def guess_column(df):
    for name in ("sentence","text","sentence1","sentence2","source","english"):
        if name in df.columns:
            return name
    return df.columns[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='datasets/cleaned/cleaned_xnli.csv')
    parser.add_argument('--output', default='datasets/translated/deepl/deepl_translated_xnli.csv')
    parser.add_argument('--column', default=None, help='Column to translate (defaults to a guessed column)')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--target-lang', default='TL', help='DeepL target language code (e.g., TL for Tagalog).')
    parser.add_argument('--split-sentences', action='store_true', help='Tokenize each input into sentences before translating and rejoin after.')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    col = args.column if args.column else guess_column(df)
    texts = df[col].astype(str).tolist()

    if args.split_sentences:
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
        except Exception:
            raise SystemExit('Install nltk (pip install nltk) to use --split-sentences')

        mapping = []
        to_translate = []
        for idx, t in enumerate(texts):
            sents = sent_tokenize(t)
            mapping.append((idx, len(sents)))
            to_translate.extend(sents)
    else:
        to_translate = texts
        mapping = None

    translated = []
    for i in tqdm(range(0, len(to_translate), args.batch_size), desc='Translating'):
        batch = to_translate[i:i+args.batch_size]
        for item in batch:
            translated_text = translate_text(item, args.target_lang)
            if translated_text is None:
                translated.append('')
            else:
                translated.append(translated_text)

    if mapping is not None:
        rebuilt = [''] * len(texts)
        pos = 0
        for idx, count in mapping:
            sents = translated[pos:pos+count]
            pos += count
            rebuilt[idx] = ' '.join(sents)
        df['deepl_translated'] = rebuilt
    else:
        df['deepl_translated'] = translated

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print('Saved:', args.output)

if __name__ == '__main__':
    main()