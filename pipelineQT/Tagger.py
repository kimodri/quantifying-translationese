import calamancy, os
import pandas as pd
from Errors import FileNameError

#TODO:
# Handle XL-Sum differenly, XL-Sum is not even here yet

class Tagger:

    def __init__(self, tag_dir="../datasets_sample/tagged/"):
        self.tag_dir = tag_dir
        if not os.path.exists(self.tag_dir):
            os.makedirs(self.tag_dir)

        self.tagger = calamancy.Tagger("tl_calamancy_md-0.2.0")
    
    @staticmethod
    def validate_filename(filename: str):
        """
        Checks if the filename starts with allowed prefixes.
        Raises FileNameError if the validation fails.
        """
        # startswith() accepts a tuple of strings to check against
        valid_prefixes = ("google", "azure", "deepl", "opus")
        
        if not filename.startswith(valid_prefixes):
            raise FileNameError(
                f"Invalid filename '{filename}'. It must start with one of: {', '.join(valid_prefixes)}"
            )
        
    def _get_sentence_form(self, text: str) -> int:
        
        data = list(self.tagger(text))
        first_index = next((i for i, (word, (pos, _)) in enumerate(data) if word == "ay" and pos == "PART"), None)

        # Logic to determine KA vs DKA
        if first_index is None:
            
            # SCENARIO A: No "ay" found
            print("Structure: Karaniwang Ayos (KA)")
            print("Reason: No inversion marker 'ay' detected.")
            return 1
            
        else:
            # SCENARIO B: "ay" found
            # We must ensure 'ay' isn't the very first word (which would be an interjection like "Ay! nauntog ako")
            if first_index > 0:
                print("Structure: Di-Karaniwang Ayos (DKA)")
                return 0
                
            else:
                print("Structure: Ambiguous (Likely KA with Interjection)")
                print("Reason: Found 'ay' but it was at the start of the sentence.")
                return 2

    def tag_bcopa(self, source, filename):

        self.validate_filename(filename)

        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        df_bcopa = pd.read_csv(source)
        df_bcopa["premise_form"] = df_bcopa["premise"].apply(self._get_sentence_form)
        df_bcopa["choice1_form"] = df_bcopa["choice1"].apply(self._get_sentence_form)
        df_bcopa["choice2_form"] = df_bcopa["choice2"].apply(self._get_sentence_form)

        df_bcopa.to_csv(destination_path, index=False)

    def tag_paws(self, source, filename):

        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        df_paws = pd.read_csv(source)
        df_paws["sentence_1_form"] = df_paws["sentence1"].apply(self._get_sentence_form)
        df_paws["sentence_2_form"] = df_paws["sentence2"].apply(self._get_sentence_form)

        df_paws.to_csv(destination_path, index=False)

    def tag_xnli(self, source, filename):

        self.validate_filename(filename)
        destination_path = os.path.join(self.tag_dir, f"{filename}.csv")
        df_xnli = pd.read_csv(source)
        df_xnli["sentence_1_form"] = df_xnli["sentence1"].apply(self._get_sentence_form)
        df_xnli["sentence_2_form"] = df_xnli["sentence2"].apply(self._get_sentence_form)

        df_xnli.to_csv(destination_path, index=False)

    