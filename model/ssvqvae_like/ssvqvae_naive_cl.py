import torch

from model.base_model import unit_test
from model.ssvqvae_like.ssvqvae import SsvqvaeConfig
from model.ssvqvae_like.ssvqvae_cl import SsvqvaeCLModel


class SsvqvaeNaiveCLModel(SsvqvaeCLModel):
    def __init__(self, config: SsvqvaeConfig):
        super().__init__(config)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        bert_emb = self.encode(self_event)
        hidden = torch.sigmoid(self.hidden_layer(bert_emb))
        soft = torch.softmax(self.predict_layer(hidden), dim=-1)
        cs_loss = self.config.loss_weight_supervised * self.cs_loss(soft, self_label, is_seen)
        vq_loss = self.config.loss_weight_vq * self.vq_loss(hidden, soft)
        vae_loss = self.config.loss_weight_vae * self.vae_loss(bert_emb, soft, is_seen)

        e_self = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neg = self.encode(neg_event)

        cl_loss = self.contrast_loss.cal_loss_pos_neg(e_self, e_pos, e_neg)
        cl_loss *= self.config.loss_weight_contrastive
        loss = cs_loss + vq_loss + vae_loss + cl_loss
        return loss


def main():
    config = SsvqvaeConfig()
    model = SsvqvaeNaiveCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
