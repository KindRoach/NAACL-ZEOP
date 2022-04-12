import json
from typing import List

from data_process.data_enum import DatasetEnum, SAMPLE_NUMBER_THRESHOLD
from data_process.data_reader.reader_util import fix_trigger
from data_process.event import Event, MAX_MENTION_LENGTH
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR


def read_events(data_set: DatasetEnum) -> List[Event]:
    with open(ROOT_DIR.joinpath(f"data/{data_set.value}/Few-Shot_ED.json"), "r", encoding="utf-8") as f:
        json_data = json.load(f)

    events = []
    for event_type, event_mentions in json_data.items():
        if len(event_mentions) > SAMPLE_NUMBER_THRESHOLD:
            for idx, mention in enumerate(event_mentions):
                trigger_idx_start = mention[2][0] - 1
                trigger_idx_end = trigger_idx_start + len(mention[1].split()) - 1
                words = mention[0].lower().split()
                event = Event(f"FewShotED_{event_type}_{idx}", event_type, (trigger_idx_start, trigger_idx_end), words[:MAX_MENTION_LENGTH])

                if not fix_trigger(event, mention[1]):
                    logger.warning("Trigger idx incorrect and could not be fixed.")
                    logger.warning(f"trigger={mention[1]}, event_trigger={event.trigger}, event={event}")
                    continue

                events.append(event)

    return events


if __name__ == '__main__':
    logger.info(f"all_data={len(read_events(DatasetEnum.FewShotED))}")
