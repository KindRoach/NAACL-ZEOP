import torch

from data_process.data_enum import DatasetEnum, DataTypeEnum
from data_process.event import EventTuple
from data_process.gen_train_data import load_event_tuples
from dataset.base_dataset import BaseDataset, unit_test
from tool.pretrain_model_helper import BertModelEnum


class ZopeDataset(BaseDataset):
    def __init__(self, data_set: DatasetEnum, data_type: DataTypeEnum, bert_model: BertModelEnum):
        super().__init__(data_set, data_type, bert_model)
        self.tuples = load_event_tuples(data_set, data_type)
        self.pattern = f"This is event about {self.encoder.tokenizer.mask_token}."

    def __getitem__(self, idx):
        event_tuple: EventTuple = self.tuples[idx]
        self_encoding = self.encoder.encode_pair(event_tuple.self, self.pattern)
        pos_encoding = self.encoder.encode_pair(event_tuple.positive, self.pattern)
        neu_encoding = self.encoder.encode_pair(event_tuple.neutral, self.pattern)
        neg_encoding = self.encoder.encode_pair(event_tuple.negative, self.pattern)

        tokens = self.encoder.tokenizer.tokenize(event_tuple.self.trigger)
        tigger_idx = self.encoder.tokenizer.convert_tokens_to_ids(tokens)[0]

        # For debug
        # self.encoder.tokenizer.convert_ids_to_tokens(self_encoding["input_ids"])

        item = {
            "self_event": self_encoding,
            "pos_event": pos_encoding,
            "neu_event": neu_encoding,
            "neg_event": neg_encoding,
            "self_event_label": torch.LongTensor([event_tuple.self.type_id]),
            "self_tigger_label": torch.LongTensor([tigger_idx]),
            "is_seen": torch.LongTensor([event_tuple.self.seen_flag]),
        }

        return item

    def __len__(self):
        return len(self.tuples)


def main():
    data_set = DatasetEnum.Ace
    dataset = ZopeDataset(data_set, DataTypeEnum.Train, data_set.bert)
    unit_test(dataset)


if __name__ == '__main__':
    main()
