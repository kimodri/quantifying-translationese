from Errors import IncorrectDatasetError, NoDatasetError, UnexpectedFileError
# from pyspark.sql import SparkSession   
# import pyspark.sql.functions as F
# from pyspark.sql.functions import rand
# from pyspark.sql.types import *  
import os, textwrap
import pandas as pd

class Processor:
    
    random_seed = 42

    def __init__(self, clean_dir='../datasets_sample/cleaned/'):
        self.clean_dir = clean_dir
        if not os.path.exists(self.clean_dir):
            os.makedirs(self.clean_dir)
        
        # Initialize Spark once
        # self.spark = (
        #     SparkSession.builder
        #     .config("spark.hadoop.io.native.lib.available", "false")
        #     .getOrCreate()
        # )

    def process(self, **kwargs):
        if not kwargs:
            raise NoDatasetError(textwrap.dedent("""
                Specify the datasets to be downloaded.
                Accepted: 'paws', 'bcopa', 'xnli', 'xlsum'
            """))
        
        # Map strings to functions
        dispatch = {
            'paws': self._clean_paws,
            'bcopa': self._clean_bcopa,
            # 'xlsum': self._clean_xlsum,
            'xnli': self._clean_xnli
        }

        for key, args in kwargs.items():
            if key not in dispatch:
                raise IncorrectDatasetError(f"{key} is not a valid dataset.")
            
            # args is likely a tuple: (path, sample_count)
            source_path = args[0]
            sample = args[1]
            
            # Call the function with the CORRECT arguments
            # Passing 'key' explicitly to match your definitions
            dispatch[key](source_path, key, sample)

    @staticmethod
    def _check_extension(source_path, expected_ext, dataset):
        file_name = source_path.split('/')[-1]
        extension = file_name.split('.')[-1]

        if extension != expected_ext:
            raise(UnexpectedFileError(
                textwrap.dedent(f"The expected file extension for {dataset} is: {expected_ext}")
            ))
        

    def _clean_paws(self, source_path, key, sample):

        self._check_extension(source_path, 'csv', 'PAWS')
        
        destination_path = os.path.join(self.clean_dir, f"{key}.csv")
        df_paws = pd.read_csv(source_path)

        df_paws['sentence1_length'] = df_paws['sentence1'].apply(len)
        df_paws['sentence2_length'] = df_paws['sentence2'].apply(len)

        # Filter
        mask = (
            (df_paws['sentence1_length'].between(21, 1999)) & 
            (df_paws['sentence2_length'].between(21, 1999))
        )
        df_paws = df_paws[mask]

        # Sample
        df_true = df_paws[df_paws['label'] == 1].sample(n=sample, random_state=self.random_seed)
        df_false = df_paws[df_paws['label'] == 0].sample(n=sample, random_state=self.random_seed)

        df_result = pd.concat([df_true, df_false], ignore_index=True)
        df_result.iloc[:, 0:4].to_csv(destination_path, index=False)


    def _clean_bcopa(self, source_path, key, sample):

        self._check_extension(source_path, 'csv', 'Balanced COPA')

        destination_path = os.path.join(self.clean_dir, f"{key}.csv")
        df_bcopa = pd.read_csv(source_path)

        df_bcopa['premise_length'] = df_bcopa['premise'].apply(len)
        df_bcopa['choice1_length'] = df_bcopa['choice1'].apply(len)
        df_bcopa['choice2_length'] = df_bcopa['choice2'].apply(len)

        # First filter
        first_filter_bcopa = (
            df_bcopa['premise_length'].between(21, 1999) &
            df_bcopa['choice1_length'].between(21, 1999) &
            df_bcopa['choice2_length'].between(21, 1999)
        )

        df_bcopa = df_bcopa[first_filter_bcopa]
        df_cause = df_bcopa[df_bcopa['question'] == 'cause'].sample(n=sample, random_state=self.random_seed)
        df_effect = df_bcopa[df_bcopa['question'] == 'effect'].sample(n=sample, random_state=self.random_seed)

        df_result = pd.concat([df_cause, df_effect], ignore_index=True)
        df_result.iloc[:, 0:7].to_csv(destination_path, index=False)
    

    # def _clean_xlsum(self, source_path, key, sample):

    #     self._check_extension(source_path, 'json', 'XL-Sum') 
    #     destination_path = os.path.join(self.clean_dir, f"{key}.csv")
    
    #     Processor._check_extension(source_path, 'tsv', 'XL-Sum')

    #     schema = StructType([
    #         StructField("text", StringType(), False),
    #         StructField("summary", StringType(), False)
    #     ])

    #     df_xlsum = self.spark.read.schema(schema).json(source_path)


    #     df_xlsum = df_xlsum.withColumn("summary_len", F.length("summary"))
    #     df_xlsum = df_xlsum.withColumn("text_len", F.length("text"))

    #     df_xlsum = df_xlsum.filter(
    #         ((F.col("summary_len") >= 20) & (F.col("text_len") >= 20)) &
    #         ((F.col("summary_len") <= 2000) & (F.col("text_len") <= 2000))
    #     )

    #     # Add a temporary column with a random number, setting a fixed seed
    #     # This step ensures the shuffle order is the same every run.
    #     df_shuffled = df_xlsum.withColumn("rand_sort_key", rand(seed=42))

    #     # Sort the DataFrame by the random key
    #     df_sorted = df_shuffled.orderBy("rand_sort_key")

    #     # Take exactly n-sample rows
    #     df_reproducible = df_sorted.limit(sample)

    #     # (Optional but good practice) Drop the temporary column
    #     df_reproducible = df_reproducible.drop("rand_sort_key", "summary_len", "text_len")

    #     pdf = df_reproducible.toPandas()
    #     pdf.to_csv(destination_path, index=False)

    def _clean_xnli(self, source_path, key, sample):

        self._check_extension(source_path, 'tsv', 'XNLI') # Fixed Label
        
        destination_path = os.path.join(self.clean_dir, f"{key}.csv")
        
        df_xnli = pd.read_csv(source_path, sep="\t")

        df_xnli = df_xnli[df_xnli["language"] == "en"]

        df_xnli = df_xnli[["gold_label", "sentence1", "sentence2"]]
        df_xnli["sentence1_len"] = df_xnli["sentence1"].apply(len)
        df_xnli["sentence2_len"] = df_xnli["sentence2"].apply(len)

        first_filter_xnli = (
            ((df_xnli['sentence1_len'] > 20) & (df_xnli['sentence1_len'] < 2000)) & 
            ((df_xnli['sentence2_len'] > 20) & (df_xnli['sentence2_len'] < 2000))
        )

        df_xnli = df_xnli[first_filter_xnli]

        df_xnli_neutral = df_xnli[df_xnli["gold_label"] == "neutral"]
        df_xnli_contradiction = df_xnli[df_xnli["gold_label"] == "contradiction"]
        df_xnli_entailment = df_xnli[df_xnli["gold_label"] == "entailment"]

        df_xnli_neutral_sample = df_xnli_neutral.sample(sample, random_state=Processor.random_seed)
        df_xnli_contradiction_sample = df_xnli_contradiction.sample(sample, random_state=Processor.random_seed)
        df_xnli_entailment_sample = df_xnli_entailment.sample(sample, random_state=Processor.random_seed)

        df_xnli_neutral_contradiction = pd.concat([df_xnli_neutral_sample, df_xnli_contradiction_sample], ignore_index=True)
        df_xnli_neutral_contadiction_entailment = pd.concat([df_xnli_neutral_contradiction, df_xnli_entailment_sample], ignore_index=True)
        df_xnli = df_xnli_neutral_contadiction_entailment.drop(columns=['sentence1_len', 'sentence2_len'])

        df_xnli.to_csv(destination_path, index=False)


def main():
    p = Processor()
    p.process(xnli=(r"C:\Users\magan\Desktop\quantifying-translationese\datasets\raw\xnli.tsv", 200))

if __name__ == "__main__":
    main()