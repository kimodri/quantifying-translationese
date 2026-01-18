import os, requests, time, textwrap
from pathlib import Path
from typing import List
import pandas as pd
from Errors import MissingKeysError, ExtraKeysError, NoDatasetError

#TODO:
# Handle XL-Sum differenly, XL-Sum is not even here yet

class Translator:

    ACCEPTED_DATASETS = ['paws', 'xnli', 'xlsum', 'bcopa']

    def __init__(self, translate_dir="../datasets_sample_translated/"):
        self.translate_dir = translate_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    @staticmethod
    def _validate_args(args, expected):
        
        invalid_items = set(args) - set(expected)

        if invalid_items:
            raise ValueError(f"Unexpected values found: {invalid_items}")

    def _google_translate(self, key, source_texts):
        
        url = "https://translation.googleapis.com/language/translate/v2"

        payload = {
            "q": source_texts,
            "target": "tl",
            "format": "text"
        }

        params = {"key": key}

        response = requests.post(url, params=params, json=payload)

        try:
            response_json = response.json()
            # print(response_json)
        except Exception:
            print("Non-JSON response:")
            print(response.text[:500])
            raise

        try:
            translations = [
                item["translatedText"]
                for item in response_json["data"]["translations"]
            ]
            return translations
        
        except Exception as e:
            print("Unexpected response structure:", e)
            print(response_json)
            raise

    def _google_batching(self, key, source_texts: list, batch_size=100):

        translated_texts = []

        for i in range(0, len(source_texts), batch_size):
            batch = source_texts[i : i + batch_size]
            batch_result = self._google_translate(key, batch)

            translated_texts.extend(batch_result)
        
        return translated_texts

    def google_translate(self, key, batch_size=100, **kwargs):
        
        if not kwargs:
            raise NoDatasetError(textwrap.dedent("""
                Specify the datasets to be translated.
                Accepted: 'paws', 'bcopa', 'xnli', 'xlsum'
            """))
        
        datasets = kwargs.keys()

        self._validate_args(datasets, Translator.ACCEPTED_DATASETS)

        if 'paws' in datasets:
            df_paws = pd.read_csv(kwargs['paws'])  # Error here if the path provided cannot be used
            sentence1 = df_paws['sentence1'].to_list()
            sentence2 = df_paws['sentence2'].to_list()

            translated_sentence1 = self._google_batching(key, sentence1, batch_size=batch_size)
            translated_sentence2 = self._google_batching(key, sentence2)

            df_paws['sentence1'] = pd.Series(translated_sentence1)
            df_paws['sentence2'] = pd.Series(translated_sentence2)

            df_paws.to_csv("../datasets_sample/translated/google_translated_paws.csv", index=False)

        if 'bcopa' in datasets:
            df_bcopa = pd.read_csv(kwargs['bcopa'])
            premise = df_bcopa['premise'].to_list()
            choice1 = df_bcopa['choice1'].to_list()
            choice2 = df_bcopa['choice2'].to_list()

            translated_premise = self._google_batching(key, premise, batch_size=batch_size)
            translated_choice1 = self._google_batching(key, choice1, batch_size=batch_size)
            translated_choice2 = self._google_batching(key, choice2, batch_size=batch_size)

            df_bcopa['premise'] = pd.Series(translated_premise)
            df_bcopa['choice1'] = pd.Series(translated_choice1)
            df_bcopa['choice2'] = pd.Series(translated_choice2)

            df_bcopa.to_csv("../datasets_sample/translated/google_translated_bcopa.csv", index=False)


        if 'xnli' in datasets:
            df_xnli = pd.read_csv(kwargs['xnli'])
            xnli_sentence1 = df_xnli["sentence1"].to_list()
            xnli_sentence2 = df_xnli["sentence2"].to_list()

            translated_sentence1 = self._google_batching(key, xnli_sentence1, batch_size=batch_size)
            translated_sentence2 = self._google_batching(key, xnli_sentence2, batch_size=batch_size)
    
            df_xnli['sentence1'] = pd.Series(translated_sentence1)
            df_xnli['sentence2'] = pd.Series(translated_sentence2)

            df_xnli.to_csv("../datasets_sample/translated/google_translated_xnli.csv", index=False)


        # TODO: XLSUM

    @staticmethod
    def _azure_translate(texts: List[str], key: str, region: str, endpoint: str) -> list:
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
    
    def _azure_batching(self, key, source_texts:list, batch_size=20):
        
        result_texts = []

        for i in range(0, len(source_texts), batch_size):
            batch = source_texts[i : i + batch_size]
            attempt = 0
            while True:
                try:
                    translated = self._azure_translate(batch, key["key"], key["region"], key["endpoint"])
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
            result_texts.extend(translated)

        return result_texts

    def azure_translate(self, key: dict, batch_size=20, **kwargs) -> list:

        if not kwargs:
            raise NoDatasetError(textwrap.dedent("""
                Specify the datasets to be translated.
                Accepted: 'paws', 'bcopa', 'xnli', 'xlsum'
            """))

        # Check if the user inputs complete credentials
        required_keys = {'key', 'region', 'endpoint'}
        missing = required_keys - key.keys()
        if missing:
            raise(MissingKeysError(
                f"Error: Missing keys: {missing}"
            ))  
        
        # Check if the given datasets are all accepted
        datasets = kwargs.keys()
        self._validate_args(datasets, Translator.ACCEPTED_DATASETS)
        
        # Translate the given datasets
        if 'paws' in datasets:
            df_paws = pd.read_csv(kwargs['paws'])
            sentence1 = df_paws['sentence1'].to_list()
            sentence2 = df_paws['sentence2'].to_list()

            translated_sentence1 = self._azure_batching(key, sentence1, batch_size=batch_size)
            translated_sentence2 = self._azure_batching(key, sentence2, batch_size=batch_size)
            
            df_paws['sentence1'] = pd.Series(translated_sentence1)
            df_paws['sentence2'] = pd.Series(translated_sentence2) 

            df_paws.to_csv("../datasets_sample/translated/azure_translated_paws.csv", index=False)

        if 'bcopa' in datasets:
            df_bcopa = pd.read_csv(kwargs['bcopa'])
            premise = df_bcopa['premise'].to_list()
            choice1 = df_bcopa['choice1'].to_list()
            choice2 = df_bcopa['choice2'].to_list()

            translated_premise = self._azure_batching(key, premise, batch_size=batch_size)
            translated_choice1 = self._azure_batching(key, choice1, batch_size=batch_size)
            translated_choice2 = self._azure_batching(key, choice2, batch_size=batch_size)

            df_bcopa['premise'] = pd.Series(translated_premise)
            df_bcopa['choice1'] = pd.Series(translated_choice1)
            df_bcopa['choice2'] = pd.Series(translated_choice2)

            df_bcopa.to_csv("../datasets_sample/translated/azure_translated_bcopa.csv", index=False)

        if 'xnli' in datasets:
            df_xnli = pd.read_csv(kwargs['xnli'])
            xnli_sentence1 = df_xnli["sentence1"].to_list()
            xnli_sentence2 = df_xnli["sentence2"].to_list()

            translated_sentence1 = self._azure_batching(key, xnli_sentence1, batch_size=batch_size)
            translated_sentence2 = self._azure_batching(key, xnli_sentence2, batch_size=batch_size)
    
            df_xnli['sentence1'] = pd.Series(translated_sentence1)
            df_xnli['sentence2'] = pd.Series(translated_sentence2)

            df_xnli.to_csv("../datasets_sample/translated/azure_translated_xnli.csv", index=False)


        # TODO: XLSUM


    def deepl_translate(self, key, **kwargs):
        pass 

    def opus_translate(self, key, **kwargs):
        pass

    