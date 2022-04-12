from typing import List

from data_process.data_enum import DatasetEnum
from data_process.event import Event


class SentenceRewriter:
    def __init__(self, data_set: DatasetEnum):
        self.data_set = data_set

    def rewrite(self, event: Event, max_pair_count: int) -> List[Event]:
        raise NotImplementedError()
