from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zope_dataset import ZopeDataset
from dataset.zope_validator_dataset import ZopeValidatorDataset
from model.base_model import BaseModel, BaseConfig, DistFuncEnum, DistFuncMap, unit_test
from model.margin_con_loss import OrderedConLoss
from validator.classify_validator import ClassifyValidator


@dataclass
class ZopeConfig(BaseConfig):
    contrastive_loss_margin: float = 1.0
    loss_weight_el: float = 1.0
    loss_weight_tl: float = 1.0
    contrastive_dist_func: DistFuncEnum = DistFuncEnum.Euclidean

    # For ZopeDecModel
    loss_weight_cluster: float = 1.0
    cluster_alpha: float = 1.0


class ZopeModel(BaseModel):
    def __init__(self, config: ZopeConfig):
        super().__init__(config, True)
        self.config = config
        self.mask_idx = 5
        self.mention_offset = 8
        self.p_dist_func = DistFuncMap[DistFuncEnum.Euclidean]
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.nll_loss = torch.nn.NLLLoss(reduction="none")
        self.center = torch.nn.Parameter(torch.randn([config.k_num, self.bert_model.config.hidden_size]))
        self.contrast_loss = OrderedConLoss(margin=config.contrastive_loss_margin, dist_func=DistFuncMap[config.contrastive_dist_func])

    def forward(self, self_event, pos_event, neu_event, neg_event, self_tigger_label, self_event_label, is_seen):
        e_self, bert_out = self.encode_internal(self_event)
        e_self_dropout = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neu = self.encode(neu_event)
        e_neg = self.encode(neg_event)

        predict_tigger = self.predict_tigger(self_event, bert_out)
        tigger_loss = self.ce_loss(predict_tigger, self_tigger_label.squeeze(1)) * is_seen.squeeze(1)
        tigger_loss *= self.config.loss_weight_tl

        p_self = self.cal_cluster_prob(e_self)
        event_loss = self.nll_loss(torch.log(p_self), self_event_label.squeeze(1)) * is_seen.squeeze(1)
        event_loss *= self.config.loss_weight_el

        cl_loss = self.contrast_loss.cal_loss_5(e_self, e_self_dropout, e_pos, e_neu, e_neg, is_seen)
        return cl_loss + event_loss + tigger_loss

    def encode(self, event_encoding: Dict[str, torch.Tensor]):
        event_vec, _ = self.encode_internal(event_encoding)
        return event_vec

    def encode_internal(self, event_encoding: Dict[str, torch.Tensor]):
        bert_out = self.bert_model(
            event_encoding["input_ids"],
            event_encoding["attention_mask"],
            event_encoding["token_type_ids"])

        cls_vector = bert_out.last_hidden_state[:, 0, :]
        mask_vector = bert_out.last_hidden_state[:, self.mask_idx, :]
        event_vec = cls_vector + mask_vector

        # event_vec shape check
        batch_size = event_encoding["input_ids"].shape[0]
        bert_hidden_dim = bert_out.last_hidden_state.shape[2]
        assert event_vec.shape == (batch_size, bert_hidden_dim)

        return event_vec, bert_out

    def predict_tigger(self, event_encoding: Dict[str, torch.Tensor], bert_out):
        prediction_scores = self.bert_cls(bert_out[0])
        logit = prediction_scores[:, self.mask_idx, :]
        mention_ids = event_encoding["input_ids"][:, self.mention_offset:]
        mention_mask = torch.zeros_like(logit, dtype=torch.long).scatter(1, mention_ids, 1)
        mention_mask[:, 0] = 0  # set [PAD] mask to False
        logit *= mention_mask
        return logit

    def cal_cluster_prob(self, bert_embedding):
        # hidden/center_batch = [batch_size, k_num, bert_hidden_size]
        hidden = bert_embedding.unsqueeze(1).repeat(1, self.config.k_num, 1)
        center_batch = self.center.unsqueeze(0).repeat(hidden.shape[0], 1, 1)

        # dist/p = [batch_size, k_num]
        dist = self.p_dist_func(hidden, center_batch)
        p = torch.softmax(-dist, dim=1)
        return p

    def classify(self, event_encoding):
        bert_out = self.encode(event_encoding)
        return self.cal_cluster_prob(bert_out)

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return ZopeDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return ClassifyValidator(self, datatype, ZopeValidatorDataset)


def main():
    config = ZopeConfig()
    model = ZopeModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
