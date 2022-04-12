from typing import Type

import numpy
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from data_process.data_enum import DataTypeEnum
from dataset.base_validator_dataset import BaseValidatorDataset
from tool.log_helper import logger
from tool.model_helper import load_model
from validator.base_validator import BaseValidator, calculate_cluster_metric, calculate_classification_metric


class KmeansValidator(BaseValidator):
    def __init__(self, model, datatype: DataTypeEnum, dataset_class: Type[BaseValidatorDataset] = BaseValidatorDataset):
        super().__init__(model, datatype, dataset_class)

    def encode_event_with_dataloader(self, events, dataloader):
        self.model.eval()
        event_vectors = []
        with torch.no_grad():
            for batch in dataloader:
                self.model.move_batch_to_device(batch)
                vector = self.model.encode(batch)
                event_vectors.append(vector)

        event_types = [e.type_id for e in events]
        event_vectors = torch.cat(event_vectors).cpu()
        return numpy.array(event_types), event_vectors.numpy()

    def eval(self):
        metrics = self.eval_unseen()
        metrics.update(self.eval_seen())
        return metrics

    def eval_unseen(self):
        actual_labels, vectors = self.encode_event(self.unseen_events)
        random_seed = self.model.config.random_seed
        random_seed = random_seed if random_seed else 2021
        kmeans = KMeans(n_clusters=self.model.config.unseen_type_num, random_state=random_seed).fit(vectors)
        result = calculate_cluster_metric(actual_labels, kmeans.labels_)
        return result

    def eval_seen(self):
        x_train, y_train = self.encode_event(self.known_events)
        x_test, y_test = self.encode_event(self.seen_events)
        neigh = KNeighborsClassifier()
        neigh.fit(x_train, y_train)
        predict_labels = neigh.predict(x_test)
        result = calculate_classification_metric(y_test, predict_labels)
        return result


def eval_trained_model(model_save_name: str, device: str = "cuda:0"):
    model = load_model(model_save_name, device)
    logger.info(model_save_name)
    logger.info(model.config)
    validator = KmeansValidator(model, DataTypeEnum.Test)
    logger.info(validator.eval())


if __name__ == '__main__':
    for model_name in []:
        eval_trained_model(model_name)
