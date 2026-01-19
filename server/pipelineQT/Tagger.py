import calamancy, os, textwrap
import pandas as pd
from .Errors import FileNameError, NoDatasetError, IncorrectDatasetError
from collections import Counter
import json
import logging  # Import logging library

class Tagger:

    def __init__(self, tag_dir="../datasets_sample/tagged/", log_file="tagger_process.log"):
        # Configure logging to write to a file and the console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # --- 2. INITIALIZATION LOGS ---
        self.tag_dir = tag_dir
        if not os.path.exists(self.tag_dir):
            self.logger.info(f"Directory {self.tag_dir} not found. Creating it.")
            os.makedirs(self.tag_dir)

        self.logger.info("Loading Calamancy model (tl_calamancy_md-0.2.0)...")
        self.nlp = calamancy.load("tl_calamancy_md-0.2.0")
        self.tagger = calamancy.Tagger("tl_calamancy_md-0.2.0")
        
        if "sentencizer" not in self.nlp.pipe_names:
            self.logger.debug("Adding 'sentencizer' to pipeline.")
            self.nlp.add_pipe("sentencizer", first=True)
            
        self.logger.info("Tagger initialized successfully.")

    def __call__(self, is_csv, **kwargs):
        self.logger.info("Starting tagging process...")
        
        if not kwargs:
            error_msg = textwrap.dedent("""
                Specify the datasets to be tagged.
                Accepted: 'paws', 'bcopa', 'xnli', 'xlsum'
            """)
            self.logger.error("No datasets provided in arguments.")
            raise NoDatasetError(error_msg)
                
        dispatch = {
            'paws': self.tag_paws,
            'bcopa': self.tag_bcopa,
            'xlsum': self.tag_xlsum,
            'xnli': self.tag_xnli
        }

        all_results = {}

        for key, args in kwargs.items():
            if key not in dispatch:
                self.logger.error(f"Invalid dataset key: {key}")
                raise IncorrectDatasetError(f"{key} is not a valid dataset.")
            
            source_path = args
            self.logger.info(f"Processing dataset '{key}' from source: {source_path}")
            
            try:
                dataset_output = dispatch[key](source_path, key, is_csv=is_csv)
                all_results[key] = dataset_output
                self.logger.info(f"Completed processing '{key}'.")
            except Exception as e:
                self.logger.exception(f"Failed to process '{key}': {e}")
                raise e

        values = list(kwargs.values())
        self.logger.debug(f"Source paths processed: {values}")
        
        try:
            value_name = values[0].split("/")[-1].split("_")[0]
            destination_path = os.path.join(self.tag_dir, f"{value_name}result.json")

            self.logger.info(f"Writing results to JSON at {destination_path}")
            with open(destination_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving JSON results: {e}")
            raise e

        self.logger.info("Tagging process completed successfully.")
        return all_results

    @staticmethod
    def get_counts(tag_list):
        counts = Counter(tag_list)
        return {
            "di_karaniwang_ayos": counts.get(0, 0),
            "karaniwang_ayos": counts.get(1, 0),
            "ambiguous": counts.get(2, 0)
        }

    def validate_filename(self, filename: str):
        # self.logger.debug(f"Validating filename: {filename}")
        pass
        # valid_prefixes = ("google", "azure", "deepl", "opus", "bing")
        # if not filename.startswith(valid_prefixes):
        #     self.logger.error(f"Validation failed for filename: {filename}")
        #     raise FileNameError(...)
        
    def _get_sentence_form(self, text: str) -> int:
        if not isinstance(text, str):
            # self.logger.warning(f"Encountered non-string text input: {text}")
            return 2 

        data = list(self.tagger(text))
        first_index = next((i for i, (word, (pos, _)) in enumerate(data) if word == "ay" and pos == "PART"), None)

        if first_index is None:
            return 1 # KA
        else:
            if first_index > 0:
                return 0 # DKA
            else:
                return 2 # Ambiguous

    def tag_bcopa(self, source, filename, is_csv=False):
        # Specific logging for file operations
        self.logger.info(f"Reading BCOPA file: {source}")
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_bcopa = pd.read_csv(source)
        self.logger.info(f"Loaded BCOPA dataframe with {len(df_bcopa)} rows.")

        df_bcopa["premise_form"] = df_bcopa["premise"].apply(self._get_sentence_form)
        df_bcopa["choice1_form"] = df_bcopa["choice1"].apply(self._get_sentence_form)
        df_bcopa["choice2_form"] = df_bcopa["choice2"].apply(self._get_sentence_form)

        if is_csv == False:
            self.logger.info(f"Saving tagged BCOPA CSV to {destination_path}")
            df_bcopa.to_csv(destination_path, index=False)

        return {
            "tags_premise": self.get_counts(df_bcopa['premise_form'].to_list()),
            "tags_choice1": self.get_counts(df_bcopa['choice1_form'].to_list()),
            "tags_choice2": self.get_counts(df_bcopa['choice2_form'].to_list()),
        }

    def tag_paws(self, source, filename, is_csv=False):
        self.logger.info(f"Reading PAWS file: {source}")
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_paws = pd.read_csv(source)
        self.logger.info(f"Loaded PAWS dataframe with {len(df_paws)} rows.")

        df_paws["sentence_1_form"] = df_paws["sentence1"].apply(self._get_sentence_form)
        df_paws["sentence_2_form"] = df_paws["sentence2"].apply(self._get_sentence_form)

        if is_csv == False:
            self.logger.info(f"Saving tagged PAWS CSV to {destination_path}")
            df_paws.to_csv(destination_path, index=False)

        return {
            "tags_sentence_1": self.get_counts(df_paws['sentence_1_form'].to_list()),
            "tags_sentence_2": self.get_counts(df_paws['sentence_2_form'].to_list())
        }
        
    def tag_xnli(self, source, filename, is_csv=False):
        self.logger.info(f"Reading XNLI file: {source}")
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_xnli = pd.read_csv(source)
        self.logger.info(f"Loaded XNLI dataframe with {len(df_xnli)} rows.")

        df_xnli["sentence_1_form"] = df_xnli["sentence1"].apply(self._get_sentence_form)
        df_xnli["sentence_2_form"] = df_xnli["sentence2"].apply(self._get_sentence_form)

        if is_csv == False:
            self.logger.info(f"Saving tagged XNLI CSV to {destination_path}")
            df_xnli.to_csv(destination_path, index=False)

        return {
            "tags_sentence_1": self.get_counts(df_xnli['sentence_1_form'].to_list()),
            "tags_sentence_2": self.get_counts(df_xnli['sentence_2_form'].to_list())
        }

    def _get_sentences_calamancy(self, text):
        if not isinstance(text, str) or not text:
            return []
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def tag_xlsum(self, source, filename, is_csv=False):
        self.logger.info(f"Reading XLSUM file: {source}")
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_xlsum = pd.read_csv(source)
        self.logger.info(f"Loaded XLSUM dataframe with {len(df_xlsum)} rows.")

        df_xlsum['sentences_list'] = df_xlsum['text'].apply(self._get_sentences_calamancy) 
        df_xlsum['summary_form'] = df_xlsum['summary'].apply(self._get_sentence_form)

        all_sentences = [sent for sublist in df_xlsum['sentences_list'] for sent in sublist]
        df_xlsum.drop(columns=['sentences_list'], inplace=True)

        tags_text = [self._get_sentence_form(sentence) for sentence in all_sentences]
        tags_summarize = df_xlsum['summary_form'].to_list()

        if is_csv == False: 
            self.logger.info(f"Saving tagged XLSUM CSV to {destination_path}")
            df_xlsum.to_csv(destination_path, index=False)
            
        return {
            "tags_text": self.get_counts(tags_text),
            "tags_summary": self.get_counts(tags_summarize)
        }

if __name__ == "__main__":
    tagger = Tagger()
    print(tagger(
        False, 
        bcopa=r"../datasets/translated/deepl/deepl_translated_bcopa.csv", 
        paws=r"../datasets/translated/deepl/deepl_translated_paws.csv", 
        xnli = r"../datasets/translated/deepl/deepl_translated_xnli.csv", 
        xlsum=r"../datasets/translated/deepl/deepl_translated_xlsum.csv"
    ))