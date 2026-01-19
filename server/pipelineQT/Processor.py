from .Errors import IncorrectDatasetError, NoDatasetError, UnexpectedFileError
import os, textwrap
import pandas as pd
import platform
import requests
from tqdm import tqdm

class Processor:
    
    random_seed = 42

    def __init__(self, clean_dir='../datasets_sample/cleaned/'):
        self.clean_dir = clean_dir
        if not os.path.exists(self.clean_dir):
            os.makedirs(self.clean_dir)
        
        # Don't initialize Spark until needed
        self.spark = None

    def _check_pyspark_requirements(self):
        """Check if PySpark and Java are available"""
        issues = []
        
        # Check PySpark
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            issues.append("PySpark is not installed. Install with: pip install pyspark>=3.2.0")
        except AttributeError:
            if platform.system() == 'Windows':
                issues.append("PySpark does not work natively on Windows. Use WSL, Linux, or macOS.")
            else:
                issues.append("PySpark installation is corrupted. Reinstall with: pip install --force-reinstall pyspark>=3.2.0")
        
        # Check Java
        import subprocess
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode != 0:
                issues.append("Java is not properly configured.")
        except FileNotFoundError:
            issues.append("Java JDK 8 or 11+ is not installed. Download from: https://adoptium.net/")
        except Exception as e:
            issues.append(f"Could not verify Java installation: {e}")
        
        return issues

    def _get_spark(self):
        """Lazy initialization of Spark session with comprehensive error handling"""
        if self.spark is None:
            # Check all requirements first
            issues = self._check_pyspark_requirements()
            
            if issues:
                error_msg = textwrap.dedent("""
                Cannot process XL-Sum dataset. The following requirements are missing:
                
                """)
                for i, issue in enumerate(issues, 1):
                    error_msg += f"{i}. {issue}\n"
                
                error_msg += textwrap.dedent("""
                
                XL-Sum Processing Requirements:
                ================================
                - Java JDK 8 or 11+ (system requirement)
                - PySpark 3.2.0+ (Python package)
                - Linux, macOS, or WSL (Windows Subsystem for Linux)
                
                Alternative Options:
                ====================
                If you cannot install these requirements:
                1. Process XL-Sum on Google Colab (free, has all requirements)
                2. Use a cloud VM (AWS, Azure, GCP)
                3. Ask a colleague with Linux/Mac to process it for you
                
                Note: PAWS, BCOPA, and XNLI work on all platforms without these requirements.
                """)
                
                raise RuntimeError(error_msg)
            
            try:
                from pyspark.sql import SparkSession
                self.spark = (
                    SparkSession.builder
                    .config("spark.hadoop.io.native.lib.available", "false")
                    .getOrCreate()
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Spark session: {e}")
        
        return self.spark

    def process(self, **kwargs):
        if not kwargs:
            raise NoDatasetError(textwrap.dedent("""
                Specify the datasets to be cleaned.
                Accepted: 'paws', 'bcopa', 'xnli', 'xlsum'
            """))
        
        # Map strings to functions
        dispatch = {
            'paws': self._clean_paws,
            'bcopa': self._clean_bcopa,
            'xlsum': self._clean_xlsum,
            'xnli': self._clean_xnli
        }

        for key, args in kwargs.items():
            if key not in dispatch:
                raise IncorrectDatasetError(f"{key} is not a valid dataset.")
            
            # Call the function with the CORRECT arguments
            # Passing 'key' explicitly to match your definitions
            dispatch[key](key, args)
    
    def get_xlsum_100(self):

        url = "https://github.com/kimodri/quantifying-translationese/releases/download/v1-dataset/cleaned_xlsum.csv"
        filename = url.split("/")[-1]
        destination_path = os.path.join(self.clean_dir, filename)

        print(f"Downloading {filename}...")
        try:
            # stream=True is important for large files so they don't load into RAM at once
            response = requests.get(url, stream=True)
            response.raise_for_status() # Errors if link is broken (404)

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 # 1KB chunks

            with open(destination_path, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
            
            print(f"Download complete: {destination_path}")
            return destination_path

        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error: {err}")
            if response.status_code == 404:
                print("Check if the Repository is Public. Private repos require authentication.")
        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def _check_extension(source_path, expected_ext, dataset):
        file_name = source_path.split('/')[-1]
        extension = file_name.split('.')[-1]

        if extension != expected_ext:
            raise(UnexpectedFileError(
                textwrap.dedent(f"The expected file extension for {dataset} is: {expected_ext}")
            ))
        
    def _clean_paws(self, key, config):

        source_path = config.get("path")
        self._check_extension(source_path, 'csv', 'PAWS')
        
        destination_path = os.path.join(self.clean_dir, f"cleaned_{key}.csv")
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
        df_true = df_paws[df_paws['label'] == 1].sample(n=config.get('true_sample'), random_state=self.random_seed)
        df_false = df_paws[df_paws['label'] == 0].sample(n=config.get('false_sample'), random_state=self.random_seed)

        df_result = pd.concat([df_true, df_false], ignore_index=True)
        df_result.iloc[:, 0:4].to_csv(destination_path, index=False)

    def _clean_bcopa(self, key, config):

        source_path = config.get("path")
        self._check_extension(source_path, 'csv', 'Balanced COPA')

        destination_path = os.path.join(self.clean_dir, f"cleaned_{key}.csv")
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
        df_cause = df_bcopa[df_bcopa['question'] == 'cause'].sample(n=config.get("cause_sample"), random_state=self.random_seed)
        df_effect = df_bcopa[df_bcopa['question'] == 'effect'].sample(n=config.get("effect_sample"), random_state=self.random_seed)

        df_result = pd.concat([df_cause, df_effect], ignore_index=True)
        df_result.iloc[:, 0:7].to_csv(destination_path, index=False)
    
    def _clean_xlsum(self, key, config):
        """
            Clean XL-Sum dataset using Polars.

            Behaviour mirrors the original PySpark version:
            - reads JSON (ndjson or JSON array)
            - keeps only rows where len(text) and len(summary) are in [20, 2000]
            - reproducible sampling using `self.random_seed`
            - writes result to {self.clean_dir}/{key}.csv
        """
        import os
        import polars as pl

        source_path = config.get("path")

        # keep same extension check as before
        self._check_extension(source_path, "jsonl", "XL-Sum")
        destination_path = os.path.join(self.clean_dir, f"cleaned_{key}.csv")

        # Robustly read NDJSON (newline-delimited) first, fallback to JSON array
        try:
            df = pl.read_ndjson(source_path)
        except Exception:
            df = pl.read_json(source_path)

        # Validate required fields
        required = {"text", "summary"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"XL-Sum JSON must contain columns {required}. Found: {df.columns}")

       # compute lengths in a version-compatible way
        str_ns = pl.col("summary").str
        if hasattr(str_ns, "len_chars"):
            summary_len_expr = pl.col("summary").str.len_chars().alias("summary_len")
            text_len_expr = pl.col("text").str.len_chars().alias("text_len")
        elif hasattr(str_ns, "lengths"):
            # older polars
            summary_len_expr = pl.col("summary").str.lengths().alias("summary_len")
            text_len_expr = pl.col("text").str.lengths().alias("text_len")
        else:
            # worst-case fallback (slower): apply Python len
            summary_len_expr = pl.col("summary").apply(lambda s: len(s) if s is not None else 0).alias("summary_len")
            text_len_expr = pl.col("text").apply(lambda s: len(s) if s is not None else 0).alias("text_len")

        df = df.with_columns([summary_len_expr, text_len_expr])
        df = df.filter(
            (pl.col("summary_len") >= 20) & (pl.col("text_len") >= 20) &
            (pl.col("summary_len") <= 2000) & (pl.col("text_len") <= 2000)
        )

        # sample reproducibly
        n = min(int(config.get("pairs_sample")), df.height)
        df_sample = df.sample(n=n, seed=self.random_seed)

        # drop helper columns once and write csv
        df_sample = df_sample.drop(["summary_len", "text_len"])
        df_sample.write_csv(destination_path)

    def _clean_xlsum_spark(self, key, config):
        """
        Clean XL-Sum dataset.
        
        Requirements:
        - Java JDK 8 or 11+
        - PySpark 3.2.0+
        - Linux, macOS, or WSL
        
        Args:
            source_path: Path to the JSON file
            key: Dataset key
            sample: Number of samples to extract
        """
        # Import PySpark only when this method is called
        import pyspark.sql.functions as F
        from pyspark.sql.functions import rand
        from pyspark.sql.types import StructType, StructField, StringType
        source_path = config.get("path")
        self._check_extension(source_path, 'json', 'XL-Sum') 
        destination_path = os.path.join(self.clean_dir, f"{key}.csv")

        schema = StructType([
            StructField("text", StringType(), False),
            StructField("summary", StringType(), False)
        ])

        spark = self._get_spark()
        df_xlsum = spark.read.schema(schema).json(source_path)

        df_xlsum = df_xlsum.withColumn("summary_len", F.length("summary"))
        df_xlsum = df_xlsum.withColumn("text_len", F.length("text"))

        df_xlsum = df_xlsum.filter(
            ((F.col("summary_len") >= 20) & (F.col("text_len") >= 20)) &
            ((F.col("summary_len") <= 2000) & (F.col("text_len") <= 2000))
        )

        # Reproducible sampling
        df_shuffled = df_xlsum.withColumn("rand_sort_key", rand(seed=self.random_seed))
        df_sorted = df_shuffled.orderBy("rand_sort_key")
        df_reproducible = df_sorted.limit(config.get("pairs_sample"))
        df_reproducible = df_reproducible.drop("rand_sort_key", "summary_len", "text_len")

        pdf = df_reproducible.toPandas()
        pdf.to_csv(destination_path, index=False)
        
        print(f"Successfully processed {config.get('pairs_sample')} samples to {destination_path}")

    def _clean_xnli(self, key, config):
        source_path = config.get("path")
        self._check_extension(source_path, 'tsv', 'XNLI')
        
        destination_path = os.path.join(self.clean_dir, f"cleaned_{key}.csv")
        
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

        df_xnli_neutral_sample = df_xnli_neutral.sample(config.get("neutral_sample"), random_state=Processor.random_seed)
        df_xnli_contradiction_sample = df_xnli_contradiction.sample(config.get("contradiction_sample"), random_state=Processor.random_seed)
        df_xnli_entailment_sample = df_xnli_entailment.sample(config.get("entailment_sample"), random_state=Processor.random_seed)

        df_xnli_neutral_contradiction = pd.concat([df_xnli_neutral_sample, df_xnli_contradiction_sample], ignore_index=True)
        df_xnli_neutral_contadiction_entailment = pd.concat([df_xnli_neutral_contradiction, df_xnli_entailment_sample], ignore_index=True)
        df_xnli = df_xnli_neutral_contadiction_entailment.drop(columns=['sentence1_len', 'sentence2_len'])

        df_xnli.to_csv(destination_path, index=False)


def main():
    p = Processor()
    # p.process(xnli=(r"C:\Users\magan\Desktop\quantifying-translationese\datasets\raw\xnli.tsv", 200))
    # p.process(xlsum=("datasets/raw/xlsum.json", 1000))  # Requires Java + PySpark + Unix
    # p.get_xlsum_100()
    # p.process(xnli=(r"C:\Users\magan\Desktop\quantifying-translationese\datasets\raw\xnli.tsv", 200))
    p.process(xlsum=(r"C:\Users\magan\Desktop\quantifying-translationese\datasets\raw\xlsum.jsonl", 100))

if __name__ == "__main__":
    main()