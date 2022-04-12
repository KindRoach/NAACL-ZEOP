from data_process.data_enum import DataTypeEnum
from tool.log_helper import logger
from tool.model_helper import load_model
from validator.classify_validator import ClassifyValidator
from validator.kmeans_validator import KmeansValidator


class ScclValidator:
    def __init__(self, model, datatype: DataTypeEnum):
        self.classify = ClassifyValidator(model, datatype)
        self.kmeans = KmeansValidator(model, datatype)

    def eval(self):
        metrics = self.classify.eval_unseen()
        metrics.update(self.kmeans.eval_seen())
        return metrics


def main(model_save_name: str, device: str = "cuda:0"):
    model = load_model(model_save_name, device)
    logger.info(model_save_name)
    logger.info(model.config)
    validator = ScclValidator(model, DataTypeEnum.Test)
    logger.info(validator.eval())


if __name__ == '__main__':
    for model_name in []:
        main(model_name)
