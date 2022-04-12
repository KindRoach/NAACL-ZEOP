from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zecl_dataset import ZeclDataset
from model.base_model import BaseModel, BaseConfig, DistFuncEnum, DistFuncMap, unit_test
from model.margin_con_loss import OrderedConLoss
from validator.classify_validator import ClassifyValidator


@dataclass
class ZeclConfig(BaseConfig):
    contrastive_loss_margin: float = 1.0
    loss_weight_supervised: float = 1.0
    contrastive_dist_func: DistFuncEnum = DistFuncEnum.Euclidean

    # For ZeclDecModel
    loss_weight_cluster: float = 1.0
    cluster_alpha: float = 1.0


class ZeclModel(BaseModel):
    def __init__(self, config: ZeclConfig):
        super().__init__(config)
        self.config = config
        self.p_dist_func = DistFuncMap[DistFuncEnum.Euclidean]
        self.nll_loss = torch.nn.NLLLoss(reduction="none")
        self.center = torch.nn.Parameter(torch.randn([config.k_num, self.bert_model.config.hidden_size]))
        self.contrast_loss = OrderedConLoss(margin=config.contrastive_loss_margin, dist_func=DistFuncMap[config.contrastive_dist_func])

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        e_self = self.encode(self_event)
        e_self_dropout = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neu = self.encode(neu_event)
        e_neg = self.encode(neg_event)

        p_self = self.cal_cluster_prob(e_self)
        s_loss = self.nll_loss(torch.log(p_self), self_label.squeeze(1)) * is_seen.squeeze(1)
        s_loss *= self.config.loss_weight_supervised

        cl_loss = self.contrast_loss.cal_loss_5(e_self, e_self_dropout, e_pos, e_neu, e_neg, is_seen)
        return s_loss + cl_loss

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
        return ZeclDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return ClassifyValidator(self, datatype)


def main():
    config = ZeclConfig()
    model = ZeclModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
