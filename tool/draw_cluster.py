import pickle
import random
from typing import List

import matplotlib.pyplot as plt
import numpy
import torch
from matplotlib.axes import Axes
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_process.data_enum import DataTypeEnum
from data_process.data_reader.main_data_reader import load_events
from dataset.base_dataset import EventMentionEncoder
from tool.log_helper import logger
from tool.model_helper import load_model
from tool.path_helper import ROOT_DIR, mkdir_parent
from validator.sccl_validator import ScclValidator


def draw_many(models: List[str], tag: str = ""):
    fig, axs = plt.subplots(1, len(models), figsize=(40, 10))
    for idx, model in enumerate(models):
        predict(model)
        draw_one(axs[idx], model)
    fig.savefig(ROOT_DIR.joinpath(f"out/draw/cluster_{tag}.pdf"))


def predict(model_save_name: str, device: str = "cuda:0"):
    out_path = ROOT_DIR.joinpath(f"out/draw/vector_pt/{model_save_name}.pt")
    mkdir_parent(out_path)
    if out_path.exists():
        return

    model = load_model(model_save_name, device)
    logger.info(model_save_name)

    model = model
    encoder = EventMentionEncoder.get_encoder(model.config.bert_model)
    events = load_events(model.config.data_set)
    unseen = events.unseen
    seen = events.seen
    # events = unseen.train + unseen.dev + unseen.test
    # events = unseen.test

    # events = seen.train + seen.dev + seen.test
    events = seen.test

    config = model.config
    pin_memory = not config.use_cpu
    validator = model.get_validator(DataTypeEnum.Test)
    if isinstance(validator, ScclValidator):
        validator = validator.classify
    dataset = validator.dataset_class(events, encoder)
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, pin_memory=pin_memory)

    model.eval()
    event_vectors = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model.move_batch_to_device(batch)
            try:
                vector = model.classify(batch)
            except NotImplementedError:
                vector = model.encode(batch)
            event_vectors.append(vector)

    event_types = numpy.array([e.type for e in events])
    event_vectors = torch.cat(event_vectors).cpu().numpy()
    event_vectors = TSNE(n_components=2).fit_transform(event_vectors)
    pickle.dump((event_types, event_vectors), open(out_path, "wb"))


def draw_one(ax: Axes, model_save_name: str):
    input_path = ROOT_DIR.joinpath(f"out/draw/vector_pt/{model_save_name}.pt")
    types, vectors = pickle.load(open(input_path, "rb"))
    vectors_by_type = dict()
    pairs = list(zip(types, vectors))
    pairs = random.Random(2021).sample(pairs, min(5000, len(types)))
    for t, v in pairs:
        if t not in vectors_by_type:
            vectors_by_type[t] = []
        vectors_by_type[t].append(v)

    for k, v in vectors_by_type.items():
        # v = random.sample(v, min(100, len(v)))
        x = [i[0] for i in v]
        y = [i[1] for i in v]
        ax.plot(x, y, "o", label=k)
    # ax.legend()
    ax.axis('off')
    ax.set_title(model_save_name.split("_")[0])


if __name__ == '__main__':
    draw_many([], tag="ACE")
    draw_many([], tag="FewShotED")
