# Reference: https://github.com/amazon-research/sccl
from typing import Dict

import torch

from model.base_model import unit_test, DistFuncMap, BaseModel
from model.margin_con_loss import OrderedConLoss
from model.sccl_like.sccl import ScclModel, ScclConfig


class ScclOrderedCLModel(ScclModel):
    def __init__(self, config: ScclConfig):
        super().__init__(config)
        self.config = config
        self.contrastive_dist_func = DistFuncMap[config.contrastive_dist_func]
        self.contrast_loss = OrderedConLoss(margin=config.contrastive_loss_margin, dist_func=DistFuncMap[config.contrastive_dist_func])

    def encode(self, event_encoding: Dict[str, torch.Tensor]):
        return BaseModel.encode(self, event_encoding)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        e_self = self.encode(self_event)
        e_self_dropout = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neu = self.encode(neu_event)
        e_neg = self.encode(neg_event)
        cl_loss = self.contrast_loss.cal_loss_5(e_self, e_self_dropout, e_pos, e_neu, e_neg, is_seen)

        # clustering loss
        cluster_p = self.cluster.cal_cluster_prob(e_self)
        cluster_loss = self.config.loss_weight_cluster * self.cluster.cal_loss(cluster_p)

        return cl_loss + cluster_loss


def main():
    config = ScclConfig()
    model = ScclOrderedCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
