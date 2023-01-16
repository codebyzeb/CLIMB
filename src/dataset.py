import os

from torch.utils.data import Dataset


class BabyDataset(Dataset):
    def __init__(self, data_root: str, sub_dir: str):
        self.data_root = data_root
        self.sub_dir = sub_dir

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
                    yield line

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return 0


def main():
    DATA_DIR = "../../../rds-personal-3CBQLhZjXbU/data/babylm_data/"
    SUB_DIR = "babylm_10M"

    BabyDataset(DATA_DIR, SUB_DIR)


if __name__ == "__main__":
    main()
