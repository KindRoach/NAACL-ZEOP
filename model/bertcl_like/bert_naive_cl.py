import torch

from model.base_model import unit_test
from model.bertcl_like.bert_cl import BertCLConfig, BertCLModel


class BertNaiveCLModel(BertCLModel):
    def __init__(self, config: BertCLConfig):
        super().__init__(config)
        self.config = config
        self.contrastive_dist_func = torch.nn.PairwiseDistance()

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        e_self = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neg = self.encode(neg_event)

        return self.contrast_loss.cal_loss_pos_neg(e_self, e_pos, e_neg)


def main():
    config = BertCLConfig()
    model = BertNaiveCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
