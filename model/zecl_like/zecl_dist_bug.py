import torch

from model.base_model import unit_test
from model.zecl_like.zecl import ZeclModel, ZeclConfig


class ZeclDistBugModel(ZeclModel):
    def __init__(self, config: ZeclConfig):
        super().__init__(config)

    def cal_cluster_prob(self, bert_embedding):
        # hidden/center_batch = [batch_size, k_num, bert_hidden_size]
        hidden = bert_embedding.unsqueeze(1).repeat(1, self.config.k_num, 1)
        center_batch = self.center.unsqueeze(0).repeat(hidden.shape[0], 1, 1)

        # dist/p = [batch_size, k_num]
        dist = self.p_dist_func(hidden, center_batch)
        p = torch.softmax(dist, dim=1)  # BUG HERE, Should be "-dist"
        return p


def main():
    config = ZeclConfig()
    model = ZeclDistBugModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
