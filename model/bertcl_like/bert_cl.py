from dataclasses import dataclass

from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.zecl_dataset import ZeclDataset
from model.base_model import BaseModel, BaseConfig, DistFuncEnum, DistFuncMap, unit_test
from model.margin_con_loss import OrderedConLoss
from validator.kmeans_validator import KmeansValidator


@dataclass
class BertCLConfig(BaseConfig):
    contrastive_loss_margin: float = 1.0
    contrastive_dist_func: DistFuncEnum = DistFuncEnum.Euclidean

    # For BertSupCLModel
    temperature: float = 0.5
    base_temperature: float = 0.07


class BertCLModel(BaseModel):
    def __init__(self, config: BertCLConfig):
        super().__init__(config)
        self.config = config
        self.contrastive_dist_func = DistFuncMap[config.contrastive_dist_func]
        self.contrast_loss = OrderedConLoss(margin=config.contrastive_loss_margin, dist_func=DistFuncMap[config.contrastive_dist_func])

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        e_self = self.encode(self_event)
        e_self_dropout = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neu = self.encode(neu_event)
        e_neg = self.encode(neg_event)

        return self.contrast_loss.cal_loss_5(e_self, e_self_dropout, e_pos, e_neu, e_neg, is_seen)

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return ZeclDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return KmeansValidator(self, datatype)


def main():
    config = BertCLConfig()
    model = BertCLModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
