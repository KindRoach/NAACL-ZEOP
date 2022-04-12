import itertools
import pickle
import random
from typing import List

import spacy
from spacy.training import Alignment
from tqdm import tqdm

from data_process.data_enum import DatasetEnum, RANDOM_STATE, DataTypeEnum
from data_process.data_reader.main_data_reader import load_events
from data_process.event import Event, EventTuple
from data_process.rewriter.translate_rewriter import TranslateRewriter
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, mkdir_parent
from tool.pretrain_model_helper import load_tokenizer

PAIR_SAMPLE_COUNT = 5
EVENT_TUPLES_PATH = "out/processed_data/%s/%s/events_tuples.pt"


def mark_mention_pos(e: Event, spacy_nlp, bert_tokenizer) -> bool:
    # 对于重写的事件，需要重新分析 token 的 pos tag

    e.bert_tokens = bert_tokenizer.tokenize(e.mention)

    try:
        doc = spacy_nlp(e.mention)
        spacy_tokens = list(doc)
        align = Alignment.from_strings([t.text for t in spacy_tokens], [t.replace("#", "") for t in e.bert_tokens])
    except ValueError as error:
        logger.warning(error)
        logger.warning(e)
        return False

    e.bert_tokens_pos = []
    accumulate_len = 0
    for l in align.y2x.lengths:
        idx = align.y2x.dataXd[accumulate_len]
        e.bert_tokens_pos.append(spacy_tokens[idx].pos_)
        accumulate_len += l

    return True


def gen_seen_train_tuples(seen_events: List[Event], unseen_events: List[Event], data_set: DatasetEnum) -> List[EventTuple]:
    seen_events = sorted(seen_events, key=lambda e: e.type)
    onetime_random = random.Random(RANDOM_STATE)
    rewriter = TranslateRewriter(data_set)

    spacy_nlp = spacy.load("en_core_web_sm")
    bert_tokenizer = load_tokenizer(data_set.bert)

    last_type = ""
    event_tuples = []
    same_type_events = []
    other_type_events = []
    for self_event in tqdm(seen_events, desc="gen_seen_train_tuples", ncols=150):
        if self_event.type != last_type:
            same_type_events = [e for e in seen_events if e.type == self_event.type]
            other_type_events = [e for e in seen_events if e.type != self_event.type] + unseen_events
            onetime_random.shuffle(same_type_events)
            onetime_random.shuffle(other_type_events)
            same_type_events_ic = itertools.cycle(same_type_events)
            other_type_events_ic = itertools.cycle(other_type_events)
            last_type = self_event.type

        for positive_event in rewriter.rewrite(self_event, 1):
            if not mark_mention_pos(positive_event, spacy_nlp, bert_tokenizer):
                continue

            # neutral_event 不能是 self_event 自己
            neutral_event = next(same_type_events_ic)
            while neutral_event.doc_id == self_event.doc_id and len(same_type_events) > 1:
                neutral_event = next(same_type_events_ic)

            # negative_event 来自不同类型的事件集合
            negative_event = next(other_type_events_ic)

            event_tuple = EventTuple(
                self=self_event,
                positive=positive_event,
                neutral=neutral_event,
                negative=negative_event
            )

        event_tuples.append(event_tuple)

    return event_tuples


def gen_unseen_train_tuples(seen_events: List[Event], unseen_events: List[Event], data_set: DatasetEnum) -> List[EventTuple]:
    rewriter = TranslateRewriter(data_set)
    onetime_random = random.Random(RANDOM_STATE)

    spacy_nlp = spacy.load("en_core_web_sm")
    bert_tokenizer = load_tokenizer(data_set.bert)

    unknown_type_events = [e for e in unseen_events]
    other_type_events = [e for e in seen_events]
    onetime_random.shuffle(unknown_type_events)
    onetime_random.shuffle(other_type_events)
    unknown_type_events = itertools.cycle(unknown_type_events)
    other_type_events = itertools.cycle(other_type_events)

    event_tuples = []
    for self_event in tqdm(unseen_events, desc="gen_unseen_train_tuples", ncols=150):
        for positive_event in rewriter.rewrite(self_event, 1):
            if not mark_mention_pos(positive_event, spacy_nlp, bert_tokenizer):
                continue

            # neutral_event 不能是 self_event 自己
            neutral_event = next(unknown_type_events)
            while neutral_event.doc_id == self_event.doc_id:
                neutral_event = next(unknown_type_events)

            # negative_event 来自不同类型的事件集合
            negative_event = next(other_type_events)

            event_tuple = EventTuple(
                self=self_event,
                positive=positive_event,
                neutral=neutral_event,
                negative=negative_event
            )

            event_tuples.append(event_tuple)

    return event_tuples


def load_event_tuples(data_set: DatasetEnum, data_type: DataTypeEnum) -> List[EventTuple]:
    data_path = ROOT_DIR.joinpath(EVENT_TUPLES_PATH % (data_set.value, data_type.value))
    if not data_path.exists():
        events = load_events(data_set)
        if data_type == DataTypeEnum.Train:
            seen_events, unseen_events = events.seen.train, events.unseen.train
        elif data_type == DataTypeEnum.Dev:
            seen_events, unseen_events = events.seen.dev, events.unseen.dev
        elif data_type == DataTypeEnum.Test:
            seen_events, unseen_events = events.seen.test, events.unseen.test
        else:
            raise Exception("Unsupported Datatype")

        seen_tuples = gen_seen_train_tuples(seen_events, unseen_events, data_set)
        unseen_tuples = gen_unseen_train_tuples(seen_events, unseen_events, data_set)
        tuples = seen_tuples + unseen_tuples
        mkdir_parent(data_path)
        with open(data_path, 'wb') as f:
            pickle.dump(tuples, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_path, 'rb') as f:
        return pickle.load(f)


def main():
    for data_set in [
        DatasetEnum.Ace,
        DatasetEnum.FewShotED,
    ]:
        train_tuples = load_event_tuples(data_set, DataTypeEnum.Train)
        dev_tuples = load_event_tuples(data_set, DataTypeEnum.Dev)
        logger.info(f"Train={len(train_tuples)}, Dev={len(dev_tuples)}")


if __name__ == '__main__':
    main()
