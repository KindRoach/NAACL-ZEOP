import torch

from data_process.data_enum import DatasetEnum, DataTypeEnum
from data_process.data_reader.main_data_reader import load_events
from dataset.base_dataset import BaseDataset, unit_test
from tool.pretrain_model_helper import BertModelEnum


class SsvqvaeDataset(BaseDataset):
    def __init__(self, data_set: DatasetEnum, data_type: DataTypeEnum, bert_model: BertModelEnum):
        super().__init__(data_set, data_type, bert_model)
        seen_events, unseen_events = self.load_events_data()
        self.events = seen_events + unseen_events

    def load_events_data(self):
        events = load_events(self.data_set)
        if self.data_type == DataTypeEnum.Train:
            seen_events, unseen_events = events.seen.train, events.unseen.train
        elif self.data_type == DataTypeEnum.Dev:
            seen_events, unseen_events = events.seen.dev, events.unseen.dev
        elif self.data_type == DataTypeEnum.Test:
            seen_events, unseen_events = events.seen.test, events.unseen.test
        else:
            raise Exception("Unsupported Datatype")
        return seen_events, unseen_events

    def __getitem__(self, idx):
        event = self.events[idx]
        encoding = self.encoder.encode_one(event)

        item = {
            "self_event": encoding,
            'self_label': torch.LongTensor([event.type_id]),
            'is_seen': torch.LongTensor([event.seen_flag]),
        }

        return item

    def __len__(self):
        return len(self.events)


def main():
    dataset = SsvqvaeDataset(DatasetEnum.FewShotED, DataTypeEnum.Dev, BertModelEnum.BERT_BASE_EN)
    unit_test(dataset)


if __name__ == '__main__':
    main()
