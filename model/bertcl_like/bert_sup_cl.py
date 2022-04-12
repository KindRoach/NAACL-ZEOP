import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zecl_dataset import ZeclDataset
from model.base_model import unit_test, BaseModel
from model.bertcl_like.bert_cl import BertCLConfig
from model.sup_con_loss import SupConLoss
from validator.kmeans_validator import KmeansValidator


class BertSupCLModel(BaseModel):
    def __init__(self, config: BertCLConfig):
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
        e_self = self.encode(self_event)
        e_pos = self.encode(pos_event)
        feat1 = F.normalize(self.head(e_self), dim=1)
        feat2 = F.normalize(self.head(e_pos), dim=1)
        return self.contrast_loss.cal_loss(feat1, feat2)

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return ZeclDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return KmeansValidator(self, datatype)


def main():
    config = BertCLConfig()
    model = BertSupCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
