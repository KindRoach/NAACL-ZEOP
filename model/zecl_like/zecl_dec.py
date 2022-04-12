import torch

from model.base_model import unit_test
from model.deep_emb_cluster import DeepEmbCluster
from model.zecl_like.zecl import ZeclConfig, ZeclModel


class ZeclDecModel(ZeclModel):
    def __init__(self, config: ZeclConfig):
        super().__init__(config)
        self.config = config
        del self.center
        del self.p_dist_func
        self.cluster = DeepEmbCluster(config.k_num, self.bert_model.config.hidden_size, config.cluster_alpha)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        e_self = self.encode(self_event)
        e_self_dropout = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neu = self.encode(neu_event)
        e_neg = self.encode(neg_event)

        p_self = self.cal_cluster_prob(e_self)
        cluster_loss = self.config.loss_weight_cluster * self.cluster.cal_loss(p_self)

        s_loss = self.nll_loss(torch.log(p_self), self_label.squeeze(1)) * is_seen.squeeze(1)
        s_loss *= self.config.loss_weight_supervised

        cl_loss = self.contrast_loss.cal_loss_5(e_self, e_self_dropout, e_pos, e_neu, e_neg, is_seen)
        return cl_loss + s_loss + cluster_loss

    def cal_cluster_prob(self, bert_embedding):
        return self.cluster.cal_cluster_prob(bert_embedding)


def main():
    config = ZeclConfig()
    model = ZeclDecModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
