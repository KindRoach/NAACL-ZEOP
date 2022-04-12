from typing import Type

import numpy
import torch

from data_process.data_enum import DataTypeEnum
from dataset.base_validator_dataset import BaseValidatorDataset
from tool.log_helper import logger
from tool.model_helper import load_model
from validator.base_validator import BaseValidator, calculate_cluster_metric, calculate_classification_metric


class ClassifyValidator(BaseValidator):
    def __init__(self, model, datatype: DataTypeEnum, dataset_class: Type[BaseValidatorDataset] = BaseValidatorDataset):
        super().__init__(model, datatype, dataset_class)

    def encode_event_with_dataloader(self, events, dataloader):
        self.model.eval()
        predict_labels = []
        with torch.no_grad():
            for batch in dataloader:
                self.model.move_batch_to_device(batch)
                soft = self.model.classify(batch)
                predict_labels.append(torch.argmax(soft, dim=-1))

        actual_labels = [e.type_id for e in events]
        predict_labels = torch.cat(predict_labels).cpu()
        return numpy.array(actual_labels), predict_labels.numpy()

    def eval(self):
        metrics = self.eval_unseen()
        metrics.update(self.eval_seen())
        return metrics

    def eval_unseen(self):
        unseen_actual, unseen_predict = self.encode_event(self.unseen_events)
        cluster_metric = calculate_cluster_metric(unseen_actual, unseen_predict)
        return cluster_metric

    def eval_seen(self):
        seen_actual, seen_predict = self.encode_event(self.seen_events)
        class_metric = calculate_classification_metric(seen_predict, seen_actual)
        return class_metric


def main(model_save_name: str, device: str = "cuda:0"):
    model = load_model(model_save_name, device)
    logger.info(model_save_name)
    logger.info(model.config)
    validator = ClassifyValidator(model, DataTypeEnum.Test)
    logger.info(validator.eval())


if __name__ == '__main__':
    for model_name in []:
        main(model_name)
