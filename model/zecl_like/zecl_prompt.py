import torch

from data_process.data_enum import LanguageEnum, DatasetEnum
from dataset.base_dataset import ENCODE_MAX_LENGTH
from model.base_model import unit_test
from model.zecl_like.zecl import ZeclModel, ZeclConfig
from tool import pretrain_model_helper
from tool.pretrain_model_helper import UNUSED_WORD_COUNT


class ZeclPromptCenterModel(ZeclModel):
    def __init__(self, config: ZeclConfig):
        super().__init__(config)

        # delete torch parameter from module
        # center matrix is not parameter in this version.
        del self.center

        lang = self.config.data_set.lang
        if lang == LanguageEnum.English:
            sentence = "This is [unused%d] news."
            self.unused_idx = 3
        elif lang == LanguageEnum.Chinese:
            sentence = "这是[unused%d]新闻。"
            self.unused_idx = 3
        else:
            raise NotImplementedError()

        if config.k_num > UNUSED_WORD_COUNT:
            raise ValueError(f"There only 99 unused word in bert-base and k is larger than {UNUSED_WORD_COUNT}.")

        tokenizer = pretrain_model_helper.load_tokenizer(config.bert_model)
        prompts = [sentence % (i + 1) for i in range(config.k_num)]
        encoding = [tokenizer(s, max_length=ENCODE_MAX_LENGTH, truncation=True, padding="max_length", return_tensors='pt') for s in prompts]
        self.center_input_ids = torch.nn.Parameter(torch.cat([e["input_ids"] for e in encoding]), False)
        self.center_attention_mask = torch.nn.Parameter(torch.cat([e["attention_mask"] for e in encoding]), False)
        self.center_token_type_ids = torch.nn.Parameter(torch.cat([e["token_type_ids"] for e in encoding]), False)

        # # for debug
        # word2id = tokenizer.get_vocab()
        # id2word = {v: k for k, v in tokenizer.get_vocab().items()}
        # sentences = [id2word[idx.item()] for idx in self.center_input_ids[0]]
        # pass

    def forward(self, **kwargs):
        bert_out = self.bert_model(self.center_input_ids, self.center_attention_mask, self.center_token_type_ids)
        self.center = bert_out.last_hidden_state[:, self.unused_idx, :]
        return super().forward(**kwargs)


def main():
    dataset = DatasetEnum.Ace
    config = ZeclConfig(data_set=dataset, bert_model=dataset.bert)
    model = ZeclPromptCenterModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
