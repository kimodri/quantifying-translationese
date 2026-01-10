#!/usr/bin/env python3
"""Translate XNLI cleaned CSV (both sentence1 and sentence2) using DeepL API."""
import argparse
import os
import time
import pandas as pd
import deepl
from tqdm import tqdm

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

def translate_batch(translator: deepl.Translator, texts: list[str], target_lang: str) -> list[str]:
    for attempt in range(3):
        try:
            result = translator.translate_text(texts, target_lang=target_lang)
            if isinstance(result, list):
                return [r.text for r in result]
            return [result.text]
        except Exception as exc:
            print(f"Translate failed (attempt {attempt + 1}/3): {exc}")
            if attempt < 2:
                time.sleep(5)
    return [""] * len(texts)

def main():
    if not DEEPL_API_KEY:
        raise SystemExit("Set DEEPL_API_KEY environment variable.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='datasets/cleaned/cleaned_xnli.csv')
    parser.add_argument('--output', default='datasets/translated/deepl/deepl_translated_xnli.csv')
    parser.add_argument('--columns', default='sentence1,sentence2')
    parser.add_argument('--target-lang', default='TL')
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()

    cols = [c.strip() for c in args.columns.split(',') if c.strip()]
    df = pd.read_csv(args.input)

    translator = deepl.Translator(DEEPL_API_KEY)
    for col in cols:
        translated: list[str] = []
        values = df[col].astype(str).tolist()
        for i in tqdm(range(0, len(values), args.batch_size), desc=f"Translating {col}"):
            batch = values[i:i+args.batch_size]
            translated.extend(translate_batch(translator, batch, args.target_lang))
        df[col] = translated

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
