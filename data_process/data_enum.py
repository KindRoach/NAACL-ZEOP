from enum import Enum

from tool.pretrain_model_helper import BertModelEnum

RANDOM_STATE = 42
SAMPLE_NUMBER_THRESHOLD = 0
MAX_MENTION_LEN = 300


class LanguageEnum(Enum):
    English = "en"
    Chinese = "zh"


class DatasetEnum(Enum):

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, lang: LanguageEnum, bert: BertModelEnum):
        self.lang = lang
        self.bert = bert

    def __str__(self):
        return str(self.value)

    Test = "TestDataset", LanguageEnum.English, BertModelEnum.BERT_BASE_EN
    Ace = "Ace", LanguageEnum.English, BertModelEnum.BERT_BASE_EN
    FewShotED = "FewShotED", LanguageEnum.English, BertModelEnum.BERT_BASE_EN


class DataTypeEnum(Enum):
    def __str__(self):
        return str(self.value)

    Train = "train"
    Test = "test"
    Dev = "dev"


def get_all_dataset():
    all_dataset = list(DatasetEnum)
    all_dataset.remove(DatasetEnum.Test)
    return all_dataset
