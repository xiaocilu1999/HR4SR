import os

import torch


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, train_loggers=None, valid_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.valid_loggers = valid_loggers if valid_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.valid_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_valid(self, log_data):
        for logger in self.valid_loggers:
            logger.log(**log_data)


class RecentModelLogger(object):
    def __init__(self, checkpoint_path, filename="checkpoint-recent.pth"):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs["epoch"]

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs["state_dict"]
            state_dict["epoch"] = kwargs["epoch"]
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(
            kwargs["state_dict"], self.checkpoint_path, self.filename + ".final"  # 如果训练被中断 可以用它继续训练
        )


class BestModelLogger(object):
    def __init__(
            self, checkpoint_path, metric_key="mean_iou", filename="best_acc_model.pth"
    ):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.0
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model at {}".format(self.metric_key, kwargs["epoch"]))
            self.best_metric = current_metric
            save_state_dict(kwargs["state_dict"], self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        pass


class MetricGraphPrinter(object):
    def __init__(
            self, writer, key="train_loss", graph_name="Train Loss", group_name="metric"
    ):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(
                self.group_name + "/" + self.graph_label,
                kwargs[self.key],
                kwargs["accumulate_iterator"],
            )
        else:
            self.writer.add_scalar(
                self.group_name + "/" + self.graph_label,
                0,
                kwargs["accumulate_iterator"],
            )

    def complete(self, *args, **kwargs):
        self.writer.close()
