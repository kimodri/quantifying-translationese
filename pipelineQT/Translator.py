import os, requests, time
from pathlib import Path
from typing import List
import pandas as pd


class Translator:
    def __init__(self, translate_dir="../datasets_sample_translated/"):
        self.translate_dir = translate_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def google_translate(self, key, **kwargs):
        pass 

    def azure_translate(self, key, **kwargs):
        pass

    @staticmethod
    def _load_azure_creds(key: str = None, region: str = None, endpoint: str = None):
        """
        Resolves Azure credentials.
        Priority:
        1. Passed arguments (variables)
        2. Environment variables (os.getenv)
        3. Local fallback file (../azure_key.txt)
        """
        # 1. Set default endpoint if not provided (Argument > Env Var > Default)
        endpoint = endpoint or os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")

        # 2. If key/region are NOT passed as arguments, try Environment Variables
        if not key:
            key = os.getenv("AZURE_TRANSLATOR_KEY")
        if not region:
            region = os.getenv("AZURE_TRANSLATOR_REGION")

        # 3. If key/region are STILL missing, look in the fallback file
        # This will only run if key or region are still None
        key_file = Path("../azure_key.txt")
        if (not key or not region) and key_file.exists():
            try:
                content = key_file.read_text()
                for raw_line in content.splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Only overwrite if we still don't have the value
                    if line.startswith("API_KEY") and not key:
                        key = line.split("=", 1)[1].strip().strip("\"' ")
                    if line.startswith("API_REGION") and not region:
                        region = line.split("=", 1)[1].strip().strip("\"' ")
            except Exception as e:
                print(f"Warning: Could not read credentials file: {e}")

        # 4. Final Validation
        if not key or not region:
            raise ValueError(
                "Missing Azure Credentials. Please provide them as arguments, "
                "set AZURE_TRANSLATOR_KEY/REGION in environment variables, "
                "or ensure ../azure_key.txt exists."
            )

        return key, region, endpoint
    
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
    







    def deepl_translate(self, key, **kwargs):
        pass 

    def opus_translate(self, key, **kwargs):
        pass

    