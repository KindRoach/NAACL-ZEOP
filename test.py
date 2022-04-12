from datetime import datetime

import pandas

from data_process.data_enum import DataTypeEnum
from tool.log_helper import logger
from tool.model_helper import load_model
from tool.path_helper import ROOT_DIR, mkdir_parent
from validator.kmeans_validator import KmeansValidator
from validator.zecl_validator import ZeclValidator


def test_model(model_save_name: str, device: str = "cuda:0"):
    model = load_model(model_save_name, device)
    logger.info(f"------------{model_save_name}------------")
    logger.info(model.config)
    validator = model.get_validator(DataTypeEnum.Test)
    logger.info(f"Using {validator.__class__.__name__}")
    if isinstance(validator, KmeansValidator) or isinstance(validator, ZeclValidator):
        result = validator.eval()
    else:
        result = validator.eval()
    logger.info(f"{result}\n\n")
    return result


def main():
    results = []
    for model_name in [
        # put model checkpoint name here (not ".pt" postfix)
        # "ZopeModel_Ace_2020_20220103004958761845",
    ]:
        try:
            result = test_model(model_name)
            result["model_save_name"] = model_name
            result["model"] = model_name.split("_")[0]
            result["dataset"] = model_name.split("_")[1]
            results.append(result)
        except FileNotFoundError:
            logger.warning(f"Model checkpoint file not found for {model_name}")

    out_path = ROOT_DIR.joinpath(f"out/eval/{datetime.now().strftime('%Y%m%d%H%M%S%f')}.csv")
    mkdir_parent(out_path)
    results = pandas.DataFrame(results)
    results.to_csv(out_path, float_format='%.4f', index=False)


if __name__ == '__main__':
    main()
