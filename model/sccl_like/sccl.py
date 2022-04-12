# Reference: https://github.com/amazon-research/sccl


from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zecl_dataset import ZeclDataset
from model.base_model import BaseModel, BaseConfig, unit_test, DistFuncEnum
from model.deep_emb_cluster import DeepEmbCluster
from model.sup_con_loss import SupConLoss
from validator.sccl_validator import ScclValidator


@dataclass
class ScclConfig(BaseConfig):
    unseen_type_num: int = 40
    seen_type_num: int = 10
    head_size: int = 128
    cluster_alpha: float = 1.0
    temperature: float = 0.5
    base_temperature: float = 0.07
    loss_weight_cluster: float = 10.0

    # For ScclOrderedCLModel
    contrastive_loss_margin: float = 1.0
    contrastive_dist_func: DistFuncEnum = DistFuncEnum.Euclidean


class ScclModel(BaseModel):
    def __init__(self, config: ScclConfig):
        super().__init__(config)
        self.config = config
        hidden_size = self.bert_model.config.hidden_size
        self.contrast_loss = SupConLoss(temperature=config.temperature, base_temperature=config.base_temperature)
        self.cluster = DeepEmbCluster(config.k_num, hidden_size, config.cluster_alpha)

        # Instance-CL head
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, 128))

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        emb_self = self.encode(self_event)
        emb_self_dropout = self.encode(self_event)
        emb_pos = self.encode(pos_event)

        # Instance-CL loss
        feat1 = F.normalize(self.head(emb_self_dropout), dim=1)
        feat2 = F.normalize(self.head(emb_pos), dim=1)
        contrastive_loss = self.contrast_loss.cal_loss(feat1, feat2)

        # clustering loss
        cluster_p = self.cluster.cal_cluster_prob(emb_self)
        cluster_loss = self.config.loss_weight_cluster * self.cluster.cal_loss(cluster_p)

        return contrastive_loss + cluster_loss

    def encode(self, event_encoding: Dict[str, torch.Tensor]):
        # mean_output = [batch_size, bert_hidden_size]
        bert_out = self.bert_model(
            event_encoding["input_ids"],
            event_encoding["attention_mask"],
            event_encoding["token_type_ids"])

        attention_mask = event_encoding["attention_mask"].unsqueeze(-1)
        mask_out = bert_out.last_hidden_state * attention_mask
        mean_output = torch.sum(mask_out, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def classify(self, event_encoding: Dict[str, torch.Tensor]):
        embeddings = self.encode(event_encoding)
        p = self.cluster.cal_cluster_prob(embeddings)
        return p

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return ZeclDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return ScclValidator(self, datatype)


def main():
    config = ScclConfig()
    model = ScclModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
