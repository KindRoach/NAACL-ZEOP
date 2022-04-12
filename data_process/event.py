from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Tuple

FILTER_CHAR = [",", ".", "(", ")", '"']
MAX_MENTION_LENGTH = 512


@dataclass
class Event:
    doc_id: str
    type: str
    trigger_idx_range: Optional[Tuple[int, int]]
    mention_words: List[str]
    type_id: Optional[int] = None
    seen_flag: Optional[bool] = None
    bert_tokens: Optional[List[str]] = None
    bert_tokens_pos: Optional[List[str]] = None
    bert_token_is_trigger: Optional[List[bool]] = None

    @property
    def trigger(self) -> str:
        s = self.trigger_idx_range[0]
        e = self.trigger_idx_range[1] + 1
        trigger = " ".join(self.mention_words[s: e])
        for c in FILTER_CHAR:
            trigger = trigger.replace(c, "")
        return trigger

    @property
    def mention(self) -> str:
        return " ".join(self.mention_words)


def set_type_id(events: List[Event], start_index: int = 0):
    """
    按出现频次的降序为事件类型编号
    """
    type_counts_dec = Counter([e.type for e in events]).most_common()
    event_types = [x[0] for x in type_counts_dec]
    type2id = {t: idx for idx, t in enumerate(event_types)}
    for event in events:
        event.type_id = type2id[event.type] + start_index


@dataclass
class EventDataList:
    train: List[Event]
    dev: List[Event]
    test: List[Event]


@dataclass
class EventData:
    seen: EventDataList
    unseen: EventDataList


@dataclass
class EventTuple:
    self: Event
    positive: Event
    neutral: Event
    negative: Event
