import os
import time
from pathlib import Path
from typing import List
import requests
import pandas as pd
from dotenv import load_dotenv


def load_azure_creds():
    """Load Azure Translator creds from .env, then fallback to azure key.txt."""
    load_dotenv()
    key = os.getenv("AZURE_TRANSLATOR_KEY")
    region = os.getenv("AZURE_TRANSLATOR_REGION")
    endpoint = os.getenv(
        "AZURE_TRANSLATOR_ENDPOINT",
        "https://api.cognitive.microsofttranslator.com",
    )

    key_file = Path("../../azure key.txt")
    if key_file.exists():
        for raw_line in key_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("API_KEY") and not key:
                key = line.split("=", 1)[1].strip().strip("\"' ")
            if line.startswith("API_REGION") and not region:
                region = line.split("=", 1)[1].strip().strip("\"' ")

    if not key or not region:
        raise ValueError("Set AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION in .env or azure key.txt")
    return key, region, endpoint


def azure_translate(texts: List[str], key: str, region: str, endpoint: str) -> List[str]:
    """Translate a list of texts to Filipino using Azure Translator."""
    url = f"{endpoint.rstrip('/')}/translate"
    params = {"api-version": "3.0", "from": "en", "to": "fil"}
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-Type": "application/json",
    }
    payload = [{"Text": text} for text in texts]

    response = requests.post(url, params=params, headers=headers, json=payload)

    try:
        response_json = response.json()
    except Exception:
        print("Non-JSON response:")
        print(response.text[:500])
        raise

    if isinstance(response_json, dict) and "error" in response_json:
        raise RuntimeError(response_json)

    try:
        translations = [item["translations"][0]["text"] for item in response_json]
        return translations
    except Exception as e:
        print("Unexpected response structure:", e)
        print(response_json)
        raise


def translate_in_chunks(texts: List[str], key: str, region: str, endpoint: str, chunk_size: int = 5) -> List[str]:
    """Translate texts in small chunks with aggressive backoff."""
    results: List[str] = []
    total_chunks = (len(texts) - 1) // chunk_size + 1
    
    for idx, start in enumerate(range(0, len(texts), chunk_size)):
        chunk = texts[start : start + chunk_size]
        attempt = 0
        max_attempts = 20
        
        while attempt < max_attempts:
            try:
                translated = azure_translate(chunk, key, region, endpoint)
                print(f"  Chunk {idx+1}/{total_chunks} OK")
                results.extend(translated)
                break
            except RuntimeError as err:
                err_text = str(err)
                if "429" in err_text or "request limits" in err_text:
                    # Very long backoff to let Azure quota reset
                    backoff = 90 + (60 * attempt)
                    print(f"  Chunk {idx+1}/{total_chunks}: rate limit; sleeping {backoff}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(backoff)
                    attempt += 1
                else:
                    raise
        
        if attempt >= max_attempts:
            raise RuntimeError(f"Exceeded max retries ({max_attempts}) for chunk {idx+1}")
    
    return results


def main():
    # Load credentials
    AZURE_TRANSLATOR_KEY, AZURE_TRANSLATOR_REGION, AZURE_TRANSLATOR_ENDPOINT = load_azure_creds()
    
    # Load data
    xlsum_path = "../../datasets/cleaned/cleaned_xlsum.csv"
    df_xlsum = pd.read_csv(xlsum_path)
    print(f"Loaded XL-Sum rows: {len(df_xlsum)}")
    
    text_list = df_xlsum["text"].to_list()
    summary_list = df_xlsum["summary"].to_list()
    
    print("\nTranslating text (100 items, chunk_size=5)...")
    text_translated = translate_in_chunks(text_list, AZURE_TRANSLATOR_KEY, AZURE_TRANSLATOR_REGION, AZURE_TRANSLATOR_ENDPOINT, chunk_size=5)
    
    print("\nTranslating summary (100 items, chunk_size=5)...")
    summary_translated = translate_in_chunks(summary_list, AZURE_TRANSLATOR_KEY, AZURE_TRANSLATOR_REGION, AZURE_TRANSLATOR_ENDPOINT, chunk_size=5)
    
    # Save output
    df_xlsum["text"] = text_translated
    df_xlsum["summary"] = summary_translated
    
    output_path = "../../datasets/translated/bing/bing_translated_xlsum.csv"
    df_xlsum.to_csv(output_path, index=False)
    print(f"\nSaved translated XL-Sum to {output_path}")
    print(f"Output shape: {df_xlsum.shape}")


if __name__ == "__main__":
    main()
