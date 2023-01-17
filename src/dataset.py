import os
import tqdm 
import logging

# TODO: use our own tokenizer -- for now just loadign in the RobertaTokenizer
from transformers import RobertaTokenizer

from torch.utils.data import Dataset

class BabyDataset(Dataset):
    def __init__(self, data_root: str, sub_dir: str, tokenizer: RobertaTokenizer):
        """
        Sets up the dataset for the BabyLM model
        """
        self.data_root = data_root
        self.sub_dir = sub_dir

        # stored as a list of dictionaries with keys
        self._data = self._load_data()

    def _load_data(self):
        """
        Load the data from the data root and sub directory
        """
        for file_name in os.listdir(
            os.path.join(self.data_root, self.sub_dir)
        ):
            file_path = os.path.join(self.data_root, self.sub_dir, file_name)
            with open(file_path, "r") as f:
                for line in f:
                    tokenized_line = self.tokenizer(line)
                    print(tokenized_line)
                    exit()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

def main():
    DATA_DIR = "../../../rds-personal-3CBQLhZjXbU/data/babylm_data"
    SUB_DIR = "babylm_10M"
    tokenizer = RobertaTokenizerFast.from_pretrained("phueb/BabyBERTa-1",
                                                 add_prefix_space=True)

    BabyDataset(DATA_DIR, SUB_DIR, tokenizer)


if __name__ == "__main__":
    main()