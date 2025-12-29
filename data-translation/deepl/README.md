# DeepL PAWS Translation

This folder contains a small script to translate PAWS (cleaned) data using the DeepL API.

Usage:

1. Set your API key:

   - Windows PowerShell:

     $env:DEEPL_API_KEY = "your_api_key_here"

2. Run the script (example):

   python data-translation/deepl/translate_paws.py \
     --input datasets/cleaned/cleaned_paws.csv \
     --output datasets/translated/deepl/deepl_translated_paws.csv \
     --batch-size 10 --target-lang TL

Notes:

- The script will try to auto-detect a suitable text column if `--column` is not provided.
- Use `--split-sentences` to tokenize long texts into sentences before translating and then rejoin them.
- Ensure `DEEPL_API_KEY` is set in your environment before running.
