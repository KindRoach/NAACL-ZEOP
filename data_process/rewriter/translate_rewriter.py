import multiprocessing
from typing import List

import numexpr

from data_process.data_enum import DatasetEnum, LanguageEnum
from data_process.data_reader.main_data_reader import load_events
from data_process.event import Event
from data_process.rewriter.sentence_rewriter import SentenceRewriter
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR

# Download model from https://www.argosopentech.com/argospm/index/
MODEL_DIR = "pretrain/argos/%s"


class TranslateRewriter(SentenceRewriter):
    def __init__(self, data_set: DatasetEnum):
        super().__init__(data_set)

        from argostranslate import package, translate

        # Enable GPU acceleration by:
        # export ARGOS_DEVICE_TYPE=cuda
        numexpr.set_num_threads(multiprocessing.cpu_count())

        for model in ["translate-zh_en-1_1.argosmodel", "translate-en_zh-1_1.argosmodel"]:
            model_path = ROOT_DIR.joinpath(MODEL_DIR % model)
            package.install_from_path(model_path.__str__())

        installed_languages = translate.get_installed_languages()
        installed_languages = {l.name: l for l in installed_languages}
        en2zh = installed_languages["English"].get_translation(installed_languages["Chinese"])
        zh2en = installed_languages["Chinese"].get_translation(installed_languages["English"])

        if data_set.lang == LanguageEnum.English:
            self.trans1 = en2zh
            self.trans2 = zh2en
        elif data_set.lang == LanguageEnum.Chinese:
            self.trans1 = zh2en
            self.trans2 = en2zh
        else:
            raise ValueError("Language not supported")

    def rewrite(self, event: Event, max_pair_count: int) -> List[Event]:
        if max_pair_count > 1:
            raise ValueError("max count should not be greater than 1")

        translation = self.trans1.translate(event.mention)
        new_mention = self.trans2.translate(translation)

        new_event = Event(
            event.doc_id + "_rewrite",
            event.type, None,
            new_mention.split(),
            event.type_id,
            event.seen_flag)
        return [new_event]


def main():
    events = load_events(DatasetEnum.Ace).unseen.train
    sr = TranslateRewriter(DatasetEnum.Ace)
    logger.info(sr.rewrite(events[0], 1))


if __name__ == '__main__':
    main()
