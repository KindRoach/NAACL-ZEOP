from typing import List, Type

import numpy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import *
from torch.utils.data import DataLoader

from data_process.data_enum import DataTypeEnum
from data_process.data_reader.main_data_reader import load_events
from data_process.event import Event
from dataset.base_dataset import EventMentionEncoder
from dataset.base_validator_dataset import BaseValidatorDataset
from tool.log_helper import logger


class BaseValidator:
    def __init__(self, model, datatype: DataTypeEnum, dataset_class: Type[BaseValidatorDataset]):
        self.model = model
        self.dataset_class = dataset_class
        self.encoder = EventMentionEncoder.get_encoder(self.model.config.bert_model)
        events = load_events(self.model.config.data_set)

        if datatype == DataTypeEnum.Dev:
            self.seen_events = events.seen.dev
            self.unseen_events = events.unseen.dev
            self.known_events = events.seen.train
        elif datatype == DataTypeEnum.Test:
            self.seen_events = events.seen.test
            self.unseen_events = events.unseen.test
            self.known_events = events.seen.train + events.seen.dev
        else:
            raise NotImplementedError()

    def encode_event(self, events):
        config = self.model.config
        pin_memory = not config.use_cpu
        dataset = self.dataset_class(events, self.encoder)
        dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, pin_memory=pin_memory)
        return self.encode_event_with_dataloader(events, dataloader)

    def encode_event_with_dataloader(self, events: List[Event], dataloader: DataLoader):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


def calculate_classification_metric(actual_labels, predict_labels, tag: str = "") -> dict:
    result = dict()
    p, r, f, _ = precision_recall_fscore_support(actual_labels, predict_labels, average='weighted', zero_division=0)
    result["acc" + tag] = accuracy_score(actual_labels, predict_labels)
    result["pre" + tag] = precision_score(actual_labels, predict_labels, average='weighted', zero_division=0)
    result["rec" + tag] = recall_score(actual_labels, predict_labels, average='weighted', zero_division=0)
    result["f1" + tag] = f1_score(actual_labels, predict_labels, average='weighted', zero_division=0)
    return result


def calculate_cluster_metric(actual_labels: numpy.ndarray, predict_labels: numpy.ndarray) -> dict:
    # 正经聚类指标
    cluster_metrics = {
        rand_score: "rand",
        adjusted_rand_score: "arand",
        normalized_mutual_info_score: "nmi",
        adjusted_mutual_info_score: "anmi",
        fowlkes_mallows_score: "fm",
        completeness_score: "comp",
        homogeneity_score: "homo",
        v_measure_score: "vm"
    }

    result = {name: metric(actual_labels, predict_labels) for metric, name in cluster_metrics.items()}

    # 通过匈牙利算法(Kuhn-Munkres or Hungarian Algorithm)
    # 获得从聚类标签到真实标签的最佳映射，再计算分类指标
    # 参考：https://github.com/XifengGuo/DEC-DA/blob/master/metrics.py
    nClass = max(actual_labels.max(), predict_labels.max()) + 1
    weight_M = numpy.zeros((nClass, nClass))
    for i in range(len(actual_labels)):
        weight_M[predict_labels[i], actual_labels[i],] += 1
    ind = linear_sum_assignment(weight_M.max() - weight_M)
    best_map = {ind[0][i]: ind[1][i] for i in range(nClass)}
    best_map_labels = [best_map[x] for x in predict_labels]
    result.update(calculate_classification_metric(actual_labels, best_map_labels, "_cluster"))

    return result


def main():
    metric = calculate_cluster_metric(numpy.array([1, 2, 3, 4, 5]), numpy.array([5, 1, 2, 3, 4]))
    logger.info(metric)
    assert metric["f1"] == 1


if __name__ == '__main__':
    main()
