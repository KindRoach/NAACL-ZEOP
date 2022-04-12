import json
from collections import Counter
from typing import List

from data_process.data_enum import DatasetEnum, SAMPLE_NUMBER_THRESHOLD
from data_process.data_reader.reader_util import fix_trigger
from data_process.event import Event, MAX_MENTION_LENGTH
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR


def read_ace_data(file_path) -> List[Event]:
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    events = []
    for e in json_data:
        if e["golden-event-mentions"]:
            event_type = e["golden-event-mentions"][0]["event_type"]
            trigger_info = e["golden-event-mentions"][0]["trigger"]
            trigger_idx_range = (trigger_info["start"], trigger_info["end"])
            words = [w.lower() for w in e["words"]]
            event = Event("", event_type, trigger_idx_range, words[:MAX_MENTION_LENGTH])

            trigger = e["golden-event-mentions"][0]["trigger"]["text"]
            if not fix_trigger(event, trigger):
                logger.warning("Trigger idx incorrect and could not be fixed.")
                logger.warning(f"trigger={trigger}, event_trigger={event.trigger}, event={event}")
                continue

            events.append(event)

    return events


def read_events(data_set: DatasetEnum) -> List[Event]:
    train_data = read_ace_data(ROOT_DIR.joinpath(f"data/{data_set.value}/train_docs.json"))
    dev_data = read_ace_data(ROOT_DIR.joinpath(f"data/{data_set.value}/dev_docs.json"))
    test_data = read_ace_data(ROOT_DIR.joinpath(f"data/{data_set.value}/test_docs.json"))

    all_data = train_data + dev_data + test_data
    type_count = Counter([e.type for e in all_data])
    all_data = [e for e in all_data if type_count[e.type] > SAMPLE_NUMBER_THRESHOLD]

    for idx, e in enumerate(all_data):
        e.doc_id = f"Ace_{idx}"

    return all_data


if __name__ == '__main__':
    logger.info(f"all_data={len(read_events(DatasetEnum.Ace))}")
