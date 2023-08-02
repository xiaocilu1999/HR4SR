import json
import os
import time
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from Logger import LoggerService, MetricGraphPrinter, RecentModelLogger, BestModelLogger
from Loss import Loss
from Metrics import Metrics
from Utils import AverageMeterSet


class Trainer(object):
    def __init__(self, args, data_loader, model, experiment_path):
        self.device = args.device
        self.model = model.to(self.device)
        self.num_epochs = args.num_epochs
        self.val_or_not = args.val_or_not
        self.best_metric = args.best_metric
        self.early_stop = args.early_stop
        self.data_loader = data_loader
        self.lr = args.lr
        self.save_noisy = None
        self.stop = None
        self.decay_step = args.decay_step
        self.weight_decay = args.weight_decay
        self.metric_ks = args.metric_ks
        self.gamma = args.gamma
        self.experiment_path = experiment_path
        self.batch_size = args.batch_size
        self.optimizer = optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        self.loss = Loss(args, self.model)
        self.metrics = Metrics(args, self.model)
        self.writer, self.train_loggers, self.valid_loggers = self.create_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.valid_loggers)

    def train(self):
        self.stop = 0
        accumulate_iterator = 0
        best_average_metrics = 0.0
        for epoch in range(1, self.num_epochs + 1):
            accumulate_iterator = self.train_one_epoch(epoch, accumulate_iterator)
            if epoch >= self.val_or_not:
                average_metrics = self.validate(epoch, accumulate_iterator)
                if average_metrics[self.best_metric] > best_average_metrics:
                    best_average_metrics = average_metrics[self.best_metric]
                    self.stop = 0
                else:
                    self.stop = self.stop + 1
                    print("Early stop: {}/{}, best {} = {}".format(self.stop, self.early_stop, self.best_metric,
                                                                   best_average_metrics))
                if self.stop >= self.early_stop:
                    print("No performance improvement for a long time, terminated.")
                    break
        self.logger_service.complete({"state_dict": (self.create_state_dict())})
        self.writer.close()
        self.test()

    def train_one_epoch(self, epoch, accumulate_iterator):
        self.model.train()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.data_loader["train_loader"])

        if self.stop == self.decay_step:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.optimizer.param_groups[0]['lr'] * self.gamma
            self.decay_step += int(self.decay_step / 2)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]
            sequence, label = batch[0], batch[1]
            self.optimizer.zero_grad()
            total_loss = self.loss.calculate_loss(sequence, label)
            total_loss.backward()
            self.optimizer.step()
            average_meter_set.update("total_loss", total_loss.item())
            tqdm_dataloader.set_description("Epoch {}, lr {:.6f}, total_loss {:.3f}".
                                            format(epoch, self.optimizer.param_groups[0]['lr'],
                                                   average_meter_set["total_loss"].average))
            accumulate_iterator += batch_size
            log_data = {
                "state_dict": (self.create_state_dict()),
                "epoch": epoch,
                "accumulate_iterator": accumulate_iterator,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_train(log_data)
        return accumulate_iterator

    def validate(self, epoch, accumulate_iterator):
        self.model.eval()
        self.save_noisy = []
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.data_loader["valid_loader"])
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                sequence, candidates, labels = batch[0], batch[1], batch[2]
                metrics, save_noisy = self.metrics.calculate_metrics(sequence, candidates, labels)
                self.save_noisy.append(save_noisy)
                for key, value in metrics.items():
                    average_meter_set.update(key, value)
                description_metrics = ["NDCG@%d" % key for key in self.metric_ks[1:2]] + ["Recall@%d" % key for key in
                                                                                          self.metric_ks[1:2]]
                description = "Valid: " + ", ".join(str_ + " {:.5f}" for str_ in description_metrics)
                description = description.format(*(average_meter_set[key].average for key in description_metrics))
                tqdm_dataloader.set_description(description)
            log_data = {
                "state_dict": (self.create_state_dict()),
                "epoch": epoch,
                "accumulate_iterator": accumulate_iterator,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_valid(log_data)

            record_metrics = ["NDCG@%d" % key for key in self.metric_ks] + ["Recall@%d" % key for key in self.metric_ks]
            record = "Valid: " + ", ".join(str_ + " {:.5f}" for str_ in record_metrics)
            record = record.format(*(average_meter_set[key].average for key in record_metrics))
            with open(os.path.join(self.experiment_path, "logs", "Training_process.txt"), "a+") as file:
                file.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " epoch " + str(
                    epoch) + ": " + record + "\n")

        return average_meter_set.averages()

    def test(self):
        print("Test best model with test set!")
        best_model = torch.load(os.path.join(self.experiment_path, "models", "best_acc_model.pth")).get(
            "model_state_dict")
        noise = torch.load(os.path.join(self.experiment_path, "models", "best_acc_model.pth")).get("noisy_state_dict")
        self.model.load_state_dict(best_model)
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.data_loader["test_loader"])
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                sequence, candidates, labels = batch[0], batch[1], batch[2]
                metrics, _ = self.metrics.calculate_metrics(sequence, candidates, labels, noise[batch_idx])
                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ["NDCG@%d" % k for k in self.metric_ks[1:2]] + ["Recall@%d" % k for k in
                                                                                      self.metric_ks[1:2]]
                description = "Test: " + ", ".join(str_ + " {:.5f}" for str_ in description_metrics)
                description = description.format(*(average_meter_set[k].average for k in description_metrics))
                tqdm_dataloader.set_description(description)
            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.experiment_path, "logs", "Test_Metrics.json"), "w") as file:
                json.dump(average_metrics, file, indent=4)
            json_formatted_str = json.dumps(average_metrics, indent=4)
            print("Test_Metrics: ", json_formatted_str.replace('"', ""))
            print('Experimental results file path: ', self.experiment_path)

    def create_loggers(self):
        root = Path(self.experiment_path)
        writer = SummaryWriter(str(root.joinpath("logs")))
        model_checkpoint = root.joinpath("models")
        train_loggers = [MetricGraphPrinter(writer, "epoch", "Epoch", "Train"),
                         MetricGraphPrinter(writer, "total_loss", "Loss", "Train")]
        valid_loggers = []
        for k in self.metric_ks:
            valid_loggers.append(MetricGraphPrinter(writer, "NDCG@%d" % k, "NDCG@%d" % k, "valid", ))
            valid_loggers.append(MetricGraphPrinter(writer, "Recall@%d" % k, "Recall@%d" % k, "valid"))
        valid_loggers.append(RecentModelLogger(model_checkpoint))
        valid_loggers.append(BestModelLogger(model_checkpoint, self.best_metric))
        return writer, train_loggers, valid_loggers

    def create_state_dict(self):
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "noisy_state_dict": self.save_noisy
        }
