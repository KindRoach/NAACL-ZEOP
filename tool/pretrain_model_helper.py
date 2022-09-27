from enum import Enum

from transformers import BertTokenizerFast, BertModel, BertForMaskedLM

from tool.path_helper import ROOT_DIR

MODEL_DIR = "pretrain/bert/%s"
UNUSED_WORD_COUNT = 999


class BertModelEnum(Enum):
    BERT_BASE_EN = "bert-base-uncased"


def load_tokenizer(bert_model: BertModelEnum):
    # download from internet
    # tokenizer = BertTokenizerFast.from_pretrained(bert_model.value)

    # use local model
    path = ROOT_DIR.joinpath(MODEL_DIR % bert_model.value).__str__()
    tokenizer = BertTokenizerFast.from_pretrained(path, local_files_only=True)
    # 把 [unused1] 加入 special_tokens，不然会被拆开
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused%d]" % (i + 1) for i in range(UNUSED_WORD_COUNT)]})
    return tokenizer


def load_bert(bert_model: BertModelEnum, for_mask: bool = False):
    model_class = BertForMaskedLM if for_mask else BertModel
    # download from internet
    # return model_class.from_pretrained(bert_model.value)

    # use local model
    path = ROOT_DIR.joinpath(MODEL_DIR % bert_model.value).__str__()
    return model_class.from_pretrained(path, local_files_only=True)
