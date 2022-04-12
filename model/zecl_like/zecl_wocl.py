import torch

from model.base_model import unit_test
from model.zecl_like.zecl import ZeclConfig, ZeclModel


class ZeclWoclModel(ZeclModel):
    def __init__(self, config: ZeclConfig):
        super().__init__(config)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        e_self = self.encode(self_event)
        p_self = self.cal_cluster_prob(e_self)
        s_loss = self.nll_loss(torch.log(p_self), self_label.squeeze(1)) * is_seen.squeeze(1)
        return s_loss


def main():
    config = ZeclConfig()
    model = ZeclWoclModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
