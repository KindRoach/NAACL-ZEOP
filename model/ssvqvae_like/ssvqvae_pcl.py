import torch

from model.base_model import unit_test
from model.ssvqvae_like.ssvqvae import SsvqvaeConfig
from model.ssvqvae_like.ssvqvae_cl import SsvqvaeCLModel


class SsvqvaeProbCLModel(SsvqvaeCLModel):
    def __init__(self, config: SsvqvaeConfig):
        super().__init__(config)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        bert_emb = self.encode(self_event)
        hidden = torch.sigmoid(self.hidden_layer(bert_emb))
        soft = torch.softmax(self.predict_layer(hidden), dim=-1)
        cs_loss = self.config.loss_weight_supervised * self.cs_loss(soft, self_label, is_seen)
        vq_loss = self.config.loss_weight_vq * self.vq_loss(hidden, soft)
        vae_loss = self.config.loss_weight_vae * self.vae_loss(bert_emb, soft, is_seen)

        p_self = self.classify(self_event)
        p_self_dropout = self.classify(self_event)
        p_pos = self.classify(pos_event)
        p_neu = self.classify(neu_event)
        p_neg = self.classify(neg_event)

        cl_loss = self.contrast_loss.cal_loss_5(p_self, p_self_dropout, p_pos, p_neu, p_neg, is_seen)
        cl_loss *= self.config.loss_weight_contrastive
        loss = cs_loss + vq_loss + vae_loss + cl_loss
        return loss


def main():
    config = SsvqvaeConfig()
    model = SsvqvaeProbCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
