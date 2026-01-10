import calamancy, os, textwrap
import pandas as pd
from Errors import FileNameError, NoDatasetError, IncorrectDatasetError
from collections import Counter

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

    def __call__(self, **kwargs):
        
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
            
            source_path = args[0]
            
            # Call the function and capture the return dictionary
            dataset_output = dispatch[key](source_path, key)
            
            # Add to the master dictionary with the dataset name as the key
            all_results[key] = dataset_output

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
        valid_prefixes = ("google", "azure", "deepl", "opus")
        if not filename.startswith(valid_prefixes):
            raise FileNameError(
                f"Invalid filename '{filename}'. It must start with one of: {', '.join(valid_prefixes)}"
            )
        
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
        df_xlsum['summarize_form'] = df_xlsum['summarize'].apply(self._get_sentence_form)

        # Get all the sentences
        all_sentences = [sent for sublist in df_xlsum['sentences_list'] for sent in sublist]

        # BUG FIX: Added columns= to drop correctly
        df_xlsum.drop(columns=['sentences_list'], inplace=True)

        tags_text = [self._get_sentence_form(sentence) for sentence in all_sentences]
        tags_summarize = df_xlsum['summarize_form'].to_list()

        if is_csv == False: # Kept consistent with other functions, assumed intentional
            df_xlsum.to_csv(destination_path, index=False)
            
        return {
            "tags_text": self.get_counts(tags_text),
            "tags_summarize": self.get_counts(tags_summarize)
        }

if __name__ == "__main__":
    pass




    # import calamancy

    # 1. Load the model
    # Ensure you have installed it first: pip install tl_calamancy_md
    # try:
    #     nlp = calamancy.load("tl_calamancy_md")
    # except OSError:
    #     print("Model not found. Please install it using: pip install tl_calamancy_md")
    #     # Alternatively, for the specific version you mentioned:
    #     # pip install https://huggingface.co/flair/tl_calamancy_md/resolve/main/tl_calamancy_md-0.1.0.tar.gz
    #     # (Note: Use the correct URL/version for your specific needs)
    #     nlp = None

    # if nlp:
    #     # 2. Input text (Tagalog paragraph)
    # nlp = calamancy.load("tl_calamancy_md-0.2.0")


    # 1. Load the model
    # nlp = calamancy.load("tl_calamancy_md")

    # --- THE FIX ---
    # We add a rule-based sentencizer at the start of the pipeline.
    # This forces splits on '.', '!', and '?' regardless of grammar predictions.
    # nlp.add_pipe("sentencizer", first=True) 
    # ----------------

    # text = "Magandang araw po, Gng. Mirabel! Ito ay isang pagsubok sa Calamancy. Sana gumana ito nang maayos. Ako ay nagt-trabaho sa D.O.H. A.E.S. naman ako nag-aral"

    # doc = nlp(text)

    # sents = [s.text for s in doc.sents]
    # print(sents)
                