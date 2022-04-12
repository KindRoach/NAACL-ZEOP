from data_process.data_enum import DataTypeEnum
from model.base_model import unit_test
from model.zecl_like.zecl import ZeclModel, ZeclConfig
from validator.kmeans_validator import KmeansValidator


class ZeclClonlyModel(ZeclModel):
    def __init__(self, config: ZeclConfig):
        super().__init__(config)

    def forward(self, self_event, pos_event, neu_event, neg_event, is_seen, self_label):
        e_self = self.encode(self_event)
        e_self_dropout = self.encode(self_event)
        e_pos = self.encode(pos_event)
        e_neu = self.encode(neu_event)
        e_neg = self.encode(neg_event)

        return self.contrast_loss.cal_loss_5(e_self, e_self_dropout, e_pos, e_neu, e_neg, is_seen)

    def get_validator(self, datatype: DataTypeEnum):
        return KmeansValidator(self, datatype)


def main():
    config = ZeclConfig()
    model = ZeclClonlyModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
