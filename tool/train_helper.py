import collections
import dataclasses
import statistics
from datetime import datetime
from itertools import repeat

import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_process.data_enum import DataTypeEnum
from model.base_model import BaseModel
from tool.log_helper import logger, add_log_file, remove_log_file
from tool.model_helper import save_model
from tool.path_helper import ROOT_DIR

REPORT_STEPS = 10


class TrainHelper:
    def __init__(self, model: BaseModel, train_tag: str):

        logger.debug("Init TrainHelper...")

        self.model = model
        self.config = dataclasses.replace(model.config)
        self.train_tag = train_tag
        self.validator = model.get_validator(DataTypeEnum.Dev)

        # init in prepare_training()
        self.epoch_length = None
        self.total_train_steps = None
        self.model_save_name = None
        self.train_dataset = None
        self.dev_dataset = None
        self.train_dataloader = None
        self.dev_dataloader = None
        self.after_epoch_dataloader = None
        self.last_save_step = None
        self.repeat_train_dataloader = None
        self.history_train_losses = None
        self.min_dev_loss = None
        self.tensorboard_writer_train = None
        self.tensorboard_writer_dev = None
        self.use_multi_gpu = None
        self.training_model = None

        # init in train_model()
        self.pbar = None

    def train_model(self):
        self.prepare_training()

        bert_params = []
        if self.model.bert_model:
            bert_params += list(self.model.bert_model.parameters())
        if self.model.bert_cls:
            bert_params += list(self.model.bert_cls.parameters())

        # both bert_model and bert_cls contains word embedding weight, so distinct here.
        bert_params_set = set(bert_params)
        bert_params = list(bert_params_set)
        other_params = [p for p in self.model.parameters() if p not in bert_params_set]
        opt = transformers.AdamW(
            [
                {"params": bert_params, "lr": self.config.bert_learning_rate},
                {"params": other_params, "lr": self.config.learning_rate}
            ],
            weight_decay=self.config.weight_decay)

        self.pbar = self.create_training_bar()
        while self.model.current_training_step < self.total_train_steps:
            self.model.current_training_step += 1

            # train one step
            self.training_model.train()
            opt.zero_grad()
            batch = next(self.repeat_train_dataloader)
            self.model.move_batch_to_device(batch)
            loss = self.training_model(**batch).mean()
            loss.backward()
            opt.step()

            self.history_train_losses.append(loss.item())
            self.pbar.update()

            if self.model.current_training_step % REPORT_STEPS == 0:
                dev_loss = statistics.mean(self.history_train_losses)
                logger.debug(f"Step {self.model.current_training_step} train loss = {dev_loss :5f}")
                self.tensorboard_writer_train.add_scalar('Loss', dev_loss, self.model.current_training_step)

            if self.model.current_training_step % self.config.check_point_step == 0:
                early_stop = self.checkpoint()
                if early_stop:
                    break

        self.finish_training()

    def prepare_training(self):

        logger.debug("Prepare Training...")

        self.model_save_name = f"{self.model.get_name()}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        if self.train_tag is not None and not self.train_tag.isspace():
            insert_idx = len(self.model.get_name())
            self.model_save_name = self.model_save_name[:insert_idx] + f"_{self.train_tag}" + self.model_save_name[insert_idx:]

        # Add log file.
        log_path = f"train/{self.model_save_name}"
        add_log_file(logger, f"{log_path}/main.log")

        # tensorboard writer
        self.tensorboard_writer_train = SummaryWriter(ROOT_DIR.joinpath(f"out/log/{log_path}/train"))
        self.tensorboard_writer_dev = SummaryWriter(ROOT_DIR.joinpath(f"out/log/{log_path}/dev"))

        # DataParallel Support: Multi GPU on one machine.
        if self.config.multi_device_ids:
            self.use_multi_gpu = True
            device_ids = []
            main_device_id = 0
            if self.config.use_cpu:
                logger.warn("Multi gpu support is not enable: Main device is set to cpu.")
                self.use_multi_gpu = False
            else:
                main_device_id = int(self.config.main_device.split(":")[1])
                if main_device_id not in self.config.multi_device_ids:
                    logger.warn("Multi gpu support is not enable: Main device is not included in multi_device_ids.")
                    self.use_multi_gpu = False

                device_count = torch.cuda.device_count()
                for device_id in self.config.multi_device_ids:
                    if device_id >= device_count:
                        logger.warn(f"Multi gpu support is limited: Device {device_id} is not available on current system.")
                    else:
                        device_ids.append(device_id)

                if len(device_ids) == 1:
                    logger.warn("Multi gpu support is not enable: Only one device is available.")
                    self.use_multi_gpu = False

            if self.use_multi_gpu:
                logger.info(f"Using multi gpu: {device_ids}")
                self.training_model = torch.nn.DataParallel(
                    module=self.model,
                    device_ids=device_ids,
                    output_device=main_device_id)
                self.config.train_batch_size *= len(device_ids)
                self.config.eval_batch_size *= len(device_ids)
            else:
                self.training_model = self.model

        logger.debug("Creating dataset...")
        pin_memory = not self.config.use_cpu
        self.train_dataset = self.model.create_dataset(DataTypeEnum.Train)
        self.dev_dataset = self.model.create_dataset(DataTypeEnum.Dev)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True, pin_memory=pin_memory)
        self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=self.config.eval_batch_size, shuffle=True, pin_memory=pin_memory)
        self.after_epoch_dataloader = DataLoader(self.train_dataset, batch_size=self.config.eval_batch_size, shuffle=True, pin_memory=pin_memory)

        # early stop only after one epoch.
        self.last_save_step = len(self.train_dataloader)

        self.repeat_train_dataloader = self.wrap_repeat_dataloader()
        self.history_train_losses = collections.deque([], maxlen=self.config.check_point_step)
        self.min_dev_loss = float("inf")
        self.epoch_length = len(self.train_dataloader)
        self.total_train_steps = self.epoch_length * self.config.epoch_num

        logger.info(f"Training {self.model_save_name}...")
        logger.info(self.config.__dict__)
        logger.info(f"Total parameter numbers = {self.model.get_param_number() / 1000000:.2f}M")
        logger.info(f"Length of epoch = {len(self.train_dataloader)} batches")
        logger.info(f"Train Random Seed = {self.config.random_seed}")

    def checkpoint(self) -> bool:
        logger.debug("Check Point reached, evaluating model on dev dataset...")

        # close train pbar for eval pbar in eval_model()
        self.pbar.close()
        dev_loss = self.eval_model()
        metrics = self.validator.eval() if self.validator is not None else {}

        train_loss = statistics.mean(self.history_train_losses)
        logger.info(f"Step {self.model.current_training_step} "
                    f"loss(train/dev) = {train_loss:5f}/{dev_loss:5f} "
                    f"Custom eval metrics = {metrics}")

        self.tensorboard_writer_dev.add_scalar('Loss', dev_loss, self.model.current_training_step)
        for m in metrics.keys():
            self.tensorboard_writer_dev.add_scalar(m, metrics[m], self.model.current_training_step)

        # save best model
        if dev_loss <= self.min_dev_loss:
            self.min_dev_loss = dev_loss
            self.last_save_step = max(self.last_save_step, self.model.current_training_step)
            save_model(self.model, self.model_save_name)

        self.pbar = self.create_training_bar()

        if self.model.current_training_step - self.last_save_step >= self.config.early_stop_check_point * self.config.check_point_step:
            logger.info(f"Early stop due to no performance promotion within {self.config.early_stop_check_point} check points.")
            return True
        else:
            return False

    def eval_model(self):
        self.training_model.eval()
        all_loss = []
        with torch.no_grad():
            for batch in tqdm(self.dev_dataloader, desc=f"Evaluating {self.model_save_name}", unit="bt", ncols=150):
                self.model.move_batch_to_device(batch)
                loss = self.training_model(**batch).mean()
                all_loss.append(loss.item())
        return statistics.mean(all_loss)

    def finish_training(self):
        self.pbar.leave = True
        self.pbar.close()
        logger.info(f"Min dev loss = {self.min_dev_loss:.5f}")
        logger.info("%s trained!" % self.model_save_name)
        remove_log_file(logger)

    def create_training_bar(self):
        pbar = tqdm(initial=self.model.current_training_step, total=self.total_train_steps,
                    desc=f"Training {self.model.get_name()}", unit="bt",
                    ncols=150, leave=False)
        if self.pbar is not None:
            pbar.start_t = self.pbar.start_t
            pbar.last_print_t = self.pbar.last_print_t

        return pbar

    def wrap_repeat_dataloader(self):
        for loader in repeat(self.train_dataloader):
            for data in loader:
                yield data
