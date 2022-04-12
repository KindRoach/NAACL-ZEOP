from data_process.event import Event, FILTER_CHAR


def fix_trigger(event: Event, trigger: str) -> bool:
    # 检查 trigger idx range 与标记的 trigger 是否正确
    # 如果不相符尝试在 mention words 中查找并修正 idx_range
    if trigger.strip() != event.trigger.strip():

        source_list = event.mention_words
        target_list = trigger.split()

        idxs = find_sub_list(source_list, target_list)
        if len(idxs) == 1:
            event.trigger_idx_range = idxs[0]

    return trigger.strip() == event.trigger.strip()


def find_sub_list(source_list, target_list):
    source_list = [filter_char(w) for w in source_list]
    target_list = [filter_char(w) for w in target_list]

    results = []
    tgt_len = len(target_list)
    sur_len = len(source_list)
    for idx in range(sur_len):
        if idx + tgt_len < sur_len and source_list[idx:idx + tgt_len] == target_list:
            results.append((idx, idx + tgt_len - 1))

    return results


def filter_char(word: str):
    for c in FILTER_CHAR:
        word = word.replace(c, "")
    return word
