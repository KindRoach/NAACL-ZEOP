from typing import Dict

import torch
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zopepl_dataset import ZopePLDataset
from dataset.zopepl_validator_dataset import ZopePLValidatorDataset
from model.base_model import unit_test
from model.zope_like.zope import ZopeConfig, ZopeModel
from validator.classify_validator import ClassifyValidator


class ZopePLModel(ZopeModel):
    def __init__(self, config: ZopeConfig):
        super().__init__(config)
        self.config = config

    def encode_internal(self, event_encoding: Dict[str, torch.Tensor]):
        bert_out = self.bert_model(
            event_encoding["input_ids"],
            event_encoding["attention_mask"],
            event_encoding["token_type_ids"])

        cls_vector = bert_out.last_hidden_state[:, 0, :]
        mask_idx = event_encoding["mask_idx"].unsqueeze(2).expand(-1, -1, bert_out.last_hidden_state.shape[2])
        mask_vector = bert_out.last_hidden_state.gather(1, mask_idx).squeeze(1)
        event_vec = cls_vector + mask_vector

        # event_vec shape check
        batch_size = event_encoding["input_ids"].shape[0]
        bert_hidden_dim = bert_out.last_hidden_state.shape[2]
        assert event_vec.shape == (batch_size, bert_hidden_dim)

        return event_vec, bert_out

    def predict_tigger(self, event_encoding: Dict[str, torch.Tensor], bert_out):
        prediction_scores = self.bert_cls(bert_out[0])
        mask_idx = event_encoding["mask_idx"].unsqueeze(2).expand(-1, -1, prediction_scores.shape[2])
        logit = prediction_scores.gather(1, mask_idx).squeeze(1)
        mention_ids = event_encoding["input_ids"]
        mention_mask = torch.zeros_like(logit, dtype=torch.long).scatter(1, mention_ids, 1)
        logit *= mention_mask
        return logit

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return ZopePLDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return ClassifyValidator(self, datatype, ZopePLValidatorDataset)


def main():
    config = ZopeConfig()
    model = ZopePLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
