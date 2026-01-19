import requests, textwrap, os
from tqdm import tqdm
from .Errors import NoDatasetError, IncorrectDatasetError


class Extractor:
    
    datasets = {
        "bcopa": "https://github.com/kimodri/quantifying-translationese/releases/download/v1-dataset/balanced_copa.csv",
        "paws": "https://github.com/kimodri/quantifying-translationese/releases/download/v1-dataset/paws.csv",
        "xlsum": "https://github.com/kimodri/quantifying-translationese/releases/download/v1-dataset/xlsum.jsonl",
        "xnli": "https://github.com/kimodri/quantifying-translationese/releases/download/v1-dataset/xnli.tsv",
    }

    def __init__(self, download_dir='../datasets_sample/raw/'):
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def extract(self, *args):
        print(f"Arguments passed: {args}")
        if len(args) < 1:
            raise NoDatasetError(textwrap.dedent(
                """
                    Specify the datasets to be downloaded:
                    Accepted:
                        - 'paws'
                        - 'bcopa'
                        - 'xnli'
                        - 'xlsum'
               """
            ))
        
        for arg in args:
            if arg not in Extractor.datasets:
                raise IncorrectDatasetError(textwrap.dedent(
                    f"""
                        {arg} is not in the accepted datasets.
                        Accepted:
                            - 'paws'
                            - 'bcopa'
                            - 'xnli'
                            - 'xlsum'
                    """
                ))
            else:
                url = Extractor.datasets.get(arg)
                filename = url.split("/")[-1]
                output_path = os.path.join(self.download_dir, filename)

                print(f"Downloading {filename}...")
                try:
                    # stream=True is important for large files so they don't load into RAM at once
                    response = requests.get(url, stream=True)
                    response.raise_for_status() # Errors if link is broken (404)

                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 # 1KB chunks

                    with open(output_path, 'wb') as file, tqdm(
                        desc=filename,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for data in response.iter_content(block_size):
                            size = file.write(data)
                            bar.update(size)
                    
                    print(f"Download complete: {output_path}")
                    # return output_path

                except requests.exceptions.HTTPError as err:
                    print(f"HTTP Error: {err}")
                    if response.status_code == 404:
                        print("Check if the Repository is Public. Private repos require authentication.")
                except Exception as e:
                    print(f"An error occurred: {e}")


def main():
    e = Extractor()
    e.extract("xnli", "paws", "bcopa", "xlsum")

if __name__ == "__main__":
    main()