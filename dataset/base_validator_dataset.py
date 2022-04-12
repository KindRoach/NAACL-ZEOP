from typing import List

from torch.utils.data import Dataset

from data_process.event import Event
from dataset.base_dataset import EventMentionEncoder


class BaseValidatorDataset(Dataset):
    def __init__(self, events: List[Event], encoder: EventMentionEncoder):
        self.events = events
        self.encoder = encoder

    def __getitem__(self, idx):
        encoding = self.encoder.encode_one(self.events[idx])
        return encoding

    def __len__(self):
        return len(self.events)
