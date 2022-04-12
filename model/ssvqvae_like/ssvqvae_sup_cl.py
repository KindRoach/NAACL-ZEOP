import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zecl_dataset import ZeclDataset
from model.base_model import unit_test
from model.ssvqvae_like.ssvqvae import SsvqvaeModel, SsvqvaeConfig
from model.sup_con_loss import SupConLoss
from validator.classify_validator import ClassifyValidator


class SsvqvaeSupCLModel(SsvqvaeModel):
    def __init__(self, config: SsvqvaeConfig):
        super().__init__(config)
        self.config = config

        # Instance-CL head
        hidden_size = self.bert_model.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, 128))

        self.contrast_loss = SupConLoss(temperature=config.temperature, base_temperature=config.base_temperature)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        bert_emb = self.encode(self_event)
        hidden = torch.sigmoid(self.hidden_layer(bert_emb))
        soft = torch.softmax(self.predict_layer(hidden), dim=-1)
        cs_loss = self.config.loss_weight_supervised * self.cs_loss(soft, self_label, is_seen)
        vq_loss = self.config.loss_weight_vq * self.vq_loss(hidden, soft)
        vae_loss = self.config.loss_weight_vae * self.vae_loss(bert_emb, soft, is_seen)

        e_self = self.encode(self_event)
        e_pos = self.encode(pos_event)

        feat1 = F.normalize(self.head(e_self), dim=1)
        feat2 = F.normalize(self.head(e_pos), dim=1)
        cl_loss = self.contrast_loss.cal_loss(feat1, feat2)
        cl_loss *= self.config.loss_weight_cluster
        loss = cs_loss + vq_loss + vae_loss + cl_loss
        return loss

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return ZeclDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return ClassifyValidator(self, datatype)


def main():
    config = SsvqvaeConfig()
    model = SsvqvaeSupCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
