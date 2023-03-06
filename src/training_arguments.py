from transformers import TrainingArguments

class CustomTrainingArguments(TrainingArguments):
    def __init__(self, sampling_strategy='None', **kwargs):
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy