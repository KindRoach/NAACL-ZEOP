import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BatchEncoding

from data_process.data_enum import DatasetEnum, DataTypeEnum
from data_process.event import Event
from data_process.gen_train_data import load_event_tuples
from tool import pretrain_model_helper
from tool.log_helper import logger
from tool.pretrain_model_helper import BertModelEnum

DATA_SAVE_PATH = "out/processed_data/%s/%s/%s.pt"


class BaseDataset(Dataset):
    def __init__(self, data_set: DatasetEnum, data_type: DataTypeEnum, bert_model: BertModelEnum):
        self.data_set = data_set
        self.data_type = data_type
        self.encoder = EventMentionEncoder.get_encoder(bert_model)


ENCODE_MAX_LENGTH = 50
ENCODER_MAP = dict()


class EventMentionEncoder:

    @staticmethod
    def get_encoder(bert_model: BertModelEnum):
        """
        线程不安全！！！！！！
        """
        if bert_model not in ENCODER_MAP:
            ENCODER_MAP[bert_model] = EventMentionEncoder(bert_model)
        return ENCODER_MAP[bert_model]

    def __init__(self, bert_model: BertModelEnum):
        self.tokenizer = pretrain_model_helper.load_tokenizer(bert_model)

    def encode_one(self, event: Event) -> BatchEncoding:
        bert_encoding = self.tokenizer(event.mention, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        mask = [1 if t == "VEB" or t == "NOUN" else 0 for t in event.bert_tokens_pos]
        mask = [0] + mask[:ENCODE_MAX_LENGTH - 1] + [0] * (ENCODE_MAX_LENGTH - len(mask) - 1)

        # 如果没有候选的 tigger，那么就用 [CLS]
        if sum(mask) == 0:
            mask[0] = 1

        for k, v in bert_encoding.data.items():
            bert_encoding[k] = v[0]

        bert_encoding["trigger_mask"] = torch.tensor(mask)
        return bert_encoding

    def encode_pair(self, event: Event, pattern: str) -> BatchEncoding:
        bert_encoding = self.tokenizer(pattern, event.mention, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        for k, v in bert_encoding.data.items():
            bert_encoding[k] = v[0]
        return bert_encoding

    def encode_pair_pl(self, event: Event, pattern: str) -> BatchEncoding:
        bert_encoding = self.tokenizer(event.mention, pattern, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt')
        for k, v in bert_encoding.data.items():
            bert_encoding[k] = v[0]

        try:
            mask_idx = bert_encoding["input_ids"].tolist().index(self.tokenizer.mask_token_id)
        except ValueError:
            mask_idx = 0
        bert_encoding["mask_idx"] = torch.LongTensor([mask_idx, ])
        return bert_encoding


def unit_test(dataset):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for k, v in next(iter(dataloader)).items():
        if isinstance(v, dict):
            for ki, vi in v.items():
                logger.info(f"{k}.{ki}={vi.shape}")
        else:
            logger.info(f"{k}={v.shape}")
    for _ in tqdm(dataloader):
        pass


def main():
    encoder = EventMentionEncoder.get_encoder(BertModelEnum.BERT_BASE_EN)
    events = load_event_tuples(DatasetEnum.Ace, DataTypeEnum.Train)
    encodings = encoder.encode_one(events[0].self)
    encodings = encoder.encode_pair(events[0].self, "This is news about [MASK].")
    pass


if __name__ == '__main__':
    main()
