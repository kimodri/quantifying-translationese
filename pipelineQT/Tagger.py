import calamancy, os, textwrap
import pandas as pd
from Errors import FileNameError, NoDatasetError, IncorrectDatasetError
from collections import Counter
import json

class Tagger:

    def __init__(self, tag_dir="../datasets_sample/tagged/"):
        self.tag_dir = tag_dir
        if not os.path.exists(self.tag_dir):
            os.makedirs(self.tag_dir)

        self.nlp = calamancy.load("tl_calamancy_md-0.2.0")
        self.tagger = calamancy.Tagger("tl_calamancy_md-0.2.0")
        
        # Moved pipeline addition here to prevent "Component already exists" error on multiple calls
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

    def __call__(self, is_csv, **kwargs):
        
        if not kwargs:
            raise NoDatasetError(textwrap.dedent("""
                Specify the datasets to be tagged.
                Accepted: 'paws', 'bcopa', 'xnli', 'xlsum'
            """))
                
        # Map strings to functions
        dispatch = {
            'paws': self.tag_paws,
            'bcopa': self.tag_bcopa,
            'xlsum': self.tag_xlsum,
            'xnli': self.tag_xnli
        }

        # Initialize the results dictionary
        all_results = {}

        for key, args in kwargs.items():
            if key not in dispatch:
                raise IncorrectDatasetError(f"{key} is not a valid dataset.")
            
            source_path = args
            
            # Call the function and capture the return dictionary
            dataset_output = dispatch[key](source_path, key, is_csv=is_csv)
            
            # Add to the master dictionary with the dataset name as the key
            all_results[key] = dataset_output

        values = list(kwargs.values())
        print(values)
        value_name = values[0].split("/")[-1].split("_")[0]
        destination_path = os.path.join(self.tag_dir, f"{value_name}result.json")

        with open(destination_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        return all_results

    @staticmethod
    def get_counts(tag_list):
        counts = Counter(tag_list)
        return {
            "di_karaniwang_ayos": counts.get(0, 0),
            "karaniwang_ayos": counts.get(1, 0),
            "ambiguous": counts.get(2, 0)
        }

    @staticmethod
    def validate_filename(filename: str):
        pass
        # valid_prefixes = ("google", "azure", "deepl", "opus", "bing")
        # if not filename.startswith(valid_prefixes):
        #     raise FileNameError(
        #         f"Invalid filename '{filename}'. It must start with one of: {', '.join(valid_prefixes)}"
        #     )
        
    def _get_sentence_form(self, text: str) -> int:
        # Handle non-string inputs (NaNs) just in case
        if not isinstance(text, str):
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
        print(f"DEBUG: The path being read is: '{source}'")
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_bcopa = pd.read_csv(source)
        df_bcopa["premise_form"] = df_bcopa["premise"].apply(self._get_sentence_form)
        df_bcopa["choice1_form"] = df_bcopa["choice1"].apply(self._get_sentence_form)
        df_bcopa["choice2_form"] = df_bcopa["choice2"].apply(self._get_sentence_form)

        if is_csv == False:
            df_bcopa.to_csv(destination_path, index=False)

        return {
            "tags_premise": self.get_counts(df_bcopa['premise_form'].to_list()),
            "tags_choice1": self.get_counts(df_bcopa['choice1_form'].to_list()),
            "tags_choice2": self.get_counts(df_bcopa['choice2_form'].to_list()),
        }

    def tag_paws(self, source, filename, is_csv=False):
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_paws = pd.read_csv(source)
        df_paws["sentence_1_form"] = df_paws["sentence1"].apply(self._get_sentence_form)
        df_paws["sentence_2_form"] = df_paws["sentence2"].apply(self._get_sentence_form)

        if is_csv == False:
            df_paws.to_csv(destination_path, index=False)

        return {
            "tags_sentence_1": self.get_counts(df_paws['sentence_1_form'].to_list()),
            "tags_sentence_2": self.get_counts(df_paws['sentence_2_form'].to_list())
        }
        
    def tag_xnli(self, source, filename, is_csv=False):
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_xnli = pd.read_csv(source)
        df_xnli["sentence_1_form"] = df_xnli["sentence1"].apply(self._get_sentence_form)
        df_xnli["sentence_2_form"] = df_xnli["sentence2"].apply(self._get_sentence_form)

        if is_csv == False:
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
        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        
        df_xlsum = pd.read_csv(source)

        df_xlsum['sentences_list'] = df_xlsum['text'].apply(self._get_sentences_calamancy) 
        
        # BUG FIX: Changed == to =
        df_xlsum['summary_form'] = df_xlsum['summary'].apply(self._get_sentence_form)

        # Get all the sentences
        all_sentences = [sent for sublist in df_xlsum['sentences_list'] for sent in sublist]

        # BUG FIX: Added columns= to drop correctly
        df_xlsum.drop(columns=['sentences_list'], inplace=True)

        tags_text = [self._get_sentence_form(sentence) for sentence in all_sentences]
        tags_summarize = df_xlsum['summary_form'].to_list()

        if is_csv == False: # Kept consistent with other functions, assumed intentional
            df_xlsum.to_csv(destination_path, index=False)
            
        return {
            "tags_text": self.get_counts(tags_text),
            "tags_summary": self.get_counts(tags_summarize)
        }

if __name__ == "__main__":
    tagger = Tagger()
    print(tagger(
        False, bcopa=r"../datasets/translated/deepl/deepl_translated_bcopa.csv", paws=r"../datasets/translated/deepl/deepl_translated_paws.csv", xnli = r"../datasets/translated/deepl/deepl_translated_xnli.csv", xlsum=r"../datasets/translated/deepl/deepl_translated_xlsum.csv"
        
    ))