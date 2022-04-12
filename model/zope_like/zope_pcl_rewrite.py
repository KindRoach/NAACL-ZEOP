import torch

from model.base_model import unit_test
from model.zope_like.zope import ZopeModel, ZopeConfig


class ZopePclRewriteModel(ZopeModel):
    def __init__(self, config: ZopeConfig):
        super().__init__(config)
        self.config = config

    def forward(self, self_event, pos_event, neu_event, neg_event, self_tigger_label, self_event_label, is_seen):
        e_self, bert_out = self.encode_internal(self_event)
        p_self = self.cal_cluster_prob(e_self)
        p_self_dropout = self.classify(self_event)
        p_pos = self.classify(pos_event)


        predict_tigger = self.predict_tigger(self_event, bert_out)
        tigger_loss = self.ce_loss(predict_tigger, self_tigger_label.squeeze(1)) * is_seen.squeeze(1)
        tigger_loss *= self.config.loss_weight_tl

        event_loss = self.nll_loss(torch.log(p_self), self_event_label.squeeze(1)) * is_seen.squeeze(1)
        event_loss *= self.config.loss_weight_el

        cl_loss = self.contrast_loss.cal_loss_3(p_self, p_self_dropout, p_pos)
        return cl_loss + event_loss + tigger_loss


def main():
    config = ZopeConfig()
    model = ZopePclRewriteModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
