import torch
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zecl_dataset import ZeclDataset
from model.base_model import DistFuncMap, unit_test
from model.margin_con_loss import OrderedConLoss
from model.ssvqvae_like.ssvqvae import SsvqvaeModel, SsvqvaeConfig
from validator.classify_validator import ClassifyValidator


class SsvqvaeCLModel(SsvqvaeModel):
    def __init__(self, config: SsvqvaeConfig):
        super().__init__(config)
        self.config = config
        self.contrastive_dist_func = DistFuncMap[config.contrastive_dist_func]
        self.contrast_loss = OrderedConLoss(margin=config.contrastive_loss_margin, dist_func=DistFuncMap[config.contrastive_dist_func])

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        bert_emb = self.encode(self_event)
        hidden = torch.sigmoid(self.hidden_layer(bert_emb))
        soft = torch.softmax(self.predict_layer(hidden), dim=-1)
        cs_loss = self.config.loss_weight_supervised * self.cs_loss(soft, self_label, is_seen)
        vq_loss = self.config.loss_weight_vq * self.vq_loss(hidden, soft)
        vae_loss = self.config.loss_weight_vae * self.vae_loss(bert_emb, soft, is_seen)

        e_self = self.encode(self_event)
        e_self_dropout = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neu = self.encode(neu_event)
        e_neg = self.encode(neg_event)

        cl_loss = self.contrast_loss.cal_loss_5(e_self, e_self_dropout, e_pos, e_neu, e_neg, is_seen)
        cl_loss *= self.config.loss_weight_contrastive
        loss = cs_loss + vq_loss + vae_loss + cl_loss
        return loss

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return ZeclDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return ClassifyValidator(self, datatype)


def main():
    config = SsvqvaeConfig()
    model = SsvqvaeCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
