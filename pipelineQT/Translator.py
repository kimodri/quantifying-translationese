import os, requests, time, textwrap
from pathlib import Path
from typing import List
import pandas as pd
from Errors import MissingKeysError, ExtraKeysError, NoDatasetError

#TODO:
# Handle XL-Sum differenly, XL-Sum is not even here yet

class Translator:
    def __init__(self, translate_dir="../datasets_sample_translated/"):
        self.translate_dir = translate_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def google_translate(self, key, **kwargs):
        pass 

    @staticmethod
    def _azure_translate(texts: List[str], key: str, region: str, endpoint: str) -> List[str]:
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
            # Bubble up API errors (e.g., rate limits) for the caller to handle
            raise RuntimeError(response_json)

        try:
            translations = [item["translations"][0]["text"] for item in response_json]
            return translations
        except Exception as e:
            print("Unexpected response structure:", e)
            print(response_json)
            raise
    
    def azure_translate(self, key: dict, chunk_size: int = 20, **kwargs) -> List[str]:

        if not kwargs:
            raise NoDatasetError(textwrap.dedent("""
                Specify the datasets to be downloaded.
                Accepted: 'paws', 'bcopa', 'xnli', 'xlsum'
            """))

        # Check if the user inputs complete credentials
        required_keys = {'key', 'region', 'endpoint'}
        missing = required_keys - key.keys()
        if missing:
            raise(MissingKeysError(
                f"Error: Missing keys: {missing}"
            ))

        # Check for extra keys
        extra = key.keys() - required_keys
        if extra:
            raise(ExtraKeysError(
                f"Error: Unexpected keys: {extra}"
            ))        
        
        # Check for the datasets
        for key in kwargs.keys():
            if key == "paws":
                pass
            elif key == "bcopa":
                pass
            elif key == "xnli":
                pass
            else:
                



        results = []

        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            attempt = 0
            while True:
                try:
                    translated = self._azure_translate(chunk, key["key"], key["region"], key["endpoint"])
                    break
                except RuntimeError as err:
                    err_text = str(err)
                    if "429" in err_text or "request limits" in err_text:
                        backoff = min(60, 5 * (attempt + 1))
                        print(f"Hit rate limit; sleeping {backoff}s then retrying (attempt {attempt + 1})...")
                        time.sleep(backoff)
                        attempt += 1
                        if attempt >= 5:
                            raise
                    else:
                        raise
            results.extend(translated)
        return results

    def deepl_translate(self, key, **kwargs):
        pass 

    def opus_translate(self, key, **kwargs):
        pass

    