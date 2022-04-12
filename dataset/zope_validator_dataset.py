from typing import List

from data_process.event import Event
from dataset.base_dataset import EventMentionEncoder
from dataset.base_validator_dataset import BaseValidatorDataset


class ZopeValidatorDataset(BaseValidatorDataset):
    def __init__(self, events: List[Event], encoder: EventMentionEncoder):
        super().__init__(events, encoder)
        self.pattern = f"This is event about {self.encoder.tokenizer.mask_token}."

    def __getitem__(self, idx):
        return self.encoder.encode_pair(self.events[idx], self.pattern)

    def __len__(self):
        return len(self.events)
