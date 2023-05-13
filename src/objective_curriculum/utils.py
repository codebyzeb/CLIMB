class StackedCollator:
    def __init__(self, collators):
        self.collators = collators

    def __call__(self, *args, **kwargs):
        batch = {}
        for collator_name, collator in self.collators.items():
            batch.update(collator(*args, **kwargs))
        return batch
