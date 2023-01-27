from typing import List

import datasets

_DESCRIPTION = """\
Dataset for the shared baby language modeling task.
The goal is to train a language model from scratch on this data which represents
roughly the amount of text and speech data a young child observes.  
"""

_HOMEPAGE = "https://babylm.github.io"

filenames = [
    "aochildes.txt",
    "bnc_spoken.txt",
    "cbt.txt",
    "children_stories.txt",
    "gutenberg.txt",
    "open_subtitles.txt",
    "qed.txt",
    "simple_wikipedia.txt",
    "switchboard.txt",
    "wikipedia.txt",
]


class BabyLM(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="strict_small",
            description="Small version of the dataset with 10M words",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="strict",
            description="Full version of the dataset with 100M words",
            version="1.0.0",
        ),
    ]

    DEFAULT_CONFIG_NAME = "strict_small"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        Returns data for different splits
        """

        if self.config.name == "strict_small":
            train_data_dir = "10M"
        else:
            train_data_dir = "100M"

        urls_to_download = {
            "train": [f"{train_data_dir}/{fn}" for fn in filenames],
            "dev": [f"dev/{fn}" for fn in filenames],
            "test": [f"test/{fn}" for fn in filenames],
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepaths": downloaded_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "dev",
                    "filepaths": downloaded_files["dev"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepaths": downloaded_files["test"],
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, split, filepaths):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # the filepaths should be a list of filepaths
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        global_idx = 0

        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    yield global_idx, {"text": row}
                    global_idx += 1
