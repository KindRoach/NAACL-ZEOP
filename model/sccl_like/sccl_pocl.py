# Reference: https://github.com/amazon-research/sccl
from typing import Dict

import torch

from model.base_model import unit_test, DistFuncMap, BaseModel
from model.margin_con_loss import OrderedConLoss
from model.sccl_like.sccl import ScclModel, ScclConfig


class ScclPCLModel(ScclModel):
    def __init__(self, config: ScclConfig):
        super().__init__(config)
        self.config = config
        self.contrastive_dist_func = DistFuncMap[config.contrastive_dist_func]
        self.contrast_loss = OrderedConLoss(margin=config.contrastive_loss_margin, dist_func=DistFuncMap[config.contrastive_dist_func])

    def encode(self, event_encoding: Dict[str, torch.Tensor]):
        return BaseModel.encode(self, event_encoding)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        p_self = self.classify(self_event)
        p_self_dropout = self.classify(self_event)
        p_pos = self.classify(pos_event)
        p_neu = self.classify(neu_event)
        p_neg = self.classify(neg_event)
        cl_loss = self.contrast_loss.cal_loss_5(p_self, p_self_dropout, p_pos, p_neu, p_neg, is_seen)

        # clustering loss
        cluster_loss = self.config.loss_weight_cluster * self.cluster.cal_loss(p_self)

        return cl_loss + cluster_loss


def main():
    config = ScclConfig()
    model = ScclPCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
