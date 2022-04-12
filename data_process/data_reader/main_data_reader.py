import pickle
from collections import Counter
from statistics import stdev, mean
from typing import List

import spacy
from sklearn.model_selection import train_test_split
from spacy.training import Alignment
from tqdm import tqdm

from data_process.data_enum import DatasetEnum, RANDOM_STATE, LanguageEnum
from data_process.data_reader import ace_reader, fewshoted_reader
from data_process.event import Event, set_type_id, EventData, EventDataList
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, mkdir_parent
from tool.pretrain_model_helper import load_tokenizer

DATASET_READER_MAP = {
    DatasetEnum.Ace: ace_reader,
    DatasetEnum.FewShotED: fewshoted_reader,
}

PROCESSED_DATA_PATH = "out/processed_data/%s/events.pt"


def load_events(data_set: DatasetEnum) -> EventData:
    if data_set.lang != LanguageEnum.English:
        raise NotImplementedError()

    data_path = ROOT_DIR.joinpath(PROCESSED_DATA_PATH % data_set.value)
    if not data_path.exists():
        all_events = DATASET_READER_MAP[data_set].read_events(data_set)

        all_events = mention_nlp_pipeline(all_events, data_set)
        seen_events, unseen_events = split_seen_unseen(all_events)

        for event in seen_events:
            event.seen_flag = True

        for event in unseen_events:
            event.seen_flag = False

        set_type_id(seen_events)
        set_type_id(unseen_events, len(set([e.type for e in seen_events])))

        train_seen, dev_seen, test_seen = train_sev_test_split(seen_events)
        train_unseen, dev_unseen, test_unseen = train_sev_test_split(unseen_events)

        mkdir_parent(data_path)
        data_to_save = EventData(
            EventDataList(train_seen, dev_seen, test_seen),
            EventDataList(train_unseen, dev_unseen, test_unseen)
        )
        with open(data_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_path, 'rb') as f:
        return pickle.load(f)


def train_sev_test_split(events):
    train_events, test_events = train_test_split(
        events, test_size=0.1, random_state=RANDOM_STATE)
    train_events, dev_events = train_test_split(
        train_events, test_size=1 / 9, random_state=RANDOM_STATE)
    return train_events, dev_events, test_events


def mention_nlp_pipeline(events: List[Event], data_set: DatasetEnum):
    processed_events = []
    spacy_nlp = spacy.load("en_core_web_sm")
    bert_tokenizer = load_tokenizer(data_set.bert)
    for e in tqdm(events, desc="nlp_pipeline", ncols=150):
        e: Event
        doc = spacy_nlp(e.mention)
        spacy_tokens = list(doc)
        e.bert_tokens = bert_tokenizer.tokenize(e.mention)

        # 首先对齐 bert token 和 spacy token，用来为前者打上 pos tag
        try:
            align = Alignment.from_strings([t.text for t in spacy_tokens], [t.replace("#", "") for t in e.bert_tokens])
        except ValueError as error:
            logger.warning(error)
            logger.warning(e)
            continue

        e.bert_tokens_pos = []
        accumulate_len = 0
        for l in align.y2x.lengths:
            idx = align.y2x.dataXd[accumulate_len]
            e.bert_tokens_pos.append(spacy_tokens[idx].pos_)
            accumulate_len += l

        # 首先对齐 bert token 和 mention words，用来为前者打上 is_trigger 标记
        try:
            align = Alignment.from_strings(e.mention_words, [t.replace("#", "") for t in e.bert_tokens])
        except ValueError as error:
            logger.warning(error)
            logger.warning(e)
            continue

        e.bert_token_is_trigger = []
        accumulate_len = 0
        for l in align.y2x.lengths:
            idx = align.y2x.dataXd[accumulate_len]
            e.bert_token_is_trigger.append(e.trigger_idx_range[0] <= idx <= e.trigger_idx_range[1])
            accumulate_len += l

        processed_events.append(e)

    return processed_events


def split_seen_unseen(all_events):
    type_count = Counter([e.type for e in all_events])
    types_count_dec = list([item[0] for item in type_count.most_common()])
    seen_types = types_count_dec[0::2]  # take types at 0,2,4...
    seen_events = [e for e in all_events if e.type in seen_types]
    unseen_events = [e for e in all_events if e.type not in seen_types]
    return seen_events, unseen_events


def get_seen_unseen_type_num(data_set: DatasetEnum) -> (int, int):
    all_events = load_events(data_set)
    seen_events = all_events.seen.train + all_events.seen.dev + all_events.seen.test
    unseen_events = all_events.unseen.train + all_events.unseen.dev + all_events.unseen.test
    return len(set([e.type for e in seen_events])), len(set([e.type for e in unseen_events]))


def main():
    data_set = DatasetEnum.FewShotED
    events = load_events(data_set)
    seens = events.seen.train + events.seen.dev + events.seen.test
    unseens = events.unseen.train + events.unseen.dev + events.unseen.test
    logger.info(f"seen={len(seens)}, unseen={len(unseens)}, total={len(seens) + len(unseens)}")
    all = seens + unseens
    types_num = Counter([t.type for t in all]).most_common()
    logger.info(f"types_num={types_num}")
    nums = [item[1] for item in types_num]
    logger.info(f"mean={mean(nums)}, stdev={stdev(nums)}")


if __name__ == '__main__':
    main()
