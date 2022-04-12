import torch

from model.base_model import DistFuncEnum, unit_test
from model.zecl_like.zecl import ZeclModel, ZeclConfig


class ZeclPclModel(ZeclModel):
    def __init__(self, config: ZeclConfig):
        super().__init__(config)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        p_self = self.classify(self_event)
        p_self_dropout = self.classify(self_event)
        p_pos = self.classify(pos_event)
        p_neu = self.classify(neu_event)
        p_neg = self.classify(neg_event)

        s_loss = self.nll_loss(torch.log(p_self), self_label.squeeze(1)) * is_seen.squeeze(1)
        s_loss *= self.config.loss_weight_supervised

        cl_loss = self.contrast_loss.cal_loss_5(p_self, p_self_dropout, p_pos, p_neu, p_neg, is_seen)
        return s_loss + cl_loss


def main():
    config = ZeclConfig(contrastive_dist_func=DistFuncEnum.Wasserstein)
    model = ZeclPclModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
