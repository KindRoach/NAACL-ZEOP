import torch

from model.base_model import unit_test
from model.ssvqvae_like.ssvqvae import SsvqvaeConfig, SsvqvaeModel


class SsvqModel(SsvqvaeModel):
    def __init__(self, config: SsvqvaeConfig):
        super().__init__(config)
        del self.vae

    def forward(self, self_event, is_seen, self_label):
        bert_emb = self.encode(self_event)
        hidden = torch.sigmoid(self.hidden_layer(bert_emb))
        soft = torch.softmax(self.predict_layer(hidden), dim=-1)
        cs_loss = self.config.loss_weight_supervised * self.cs_loss(soft, self_label, is_seen)
        vq_loss = self.config.loss_weight_vq * self.vq_loss(hidden, soft)
        loss = cs_loss + vq_loss
        return loss


def main():
    config = SsvqvaeConfig(main_device="cuda:0", train_batch_size=16)
    model = SsvqModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
