import json
import os

import torch
from tqdm import tqdm

from Metrics import Metrics
from Utils import AverageMeterSet


class Inference(object):
    def __init__(self, args, model, model_path, data_loader):
        self.device = args.device
        self.metric_ks = args.metric_ks
        self.model = model.to(self.device)
        self.model_path = model_path
        self.data_loader = data_loader
        self.metrics = Metrics(args, self.model)

    def inference(self):
        print("Test best model with test set!")
        best_model = torch.load(os.path.join(self.model_path, "models", "best_acc_model.pth")).get("model_state_dict")
        noise = torch.load(os.path.join(self.model_path, "models", "best_acc_model.pth")).get("noisy_state_dict")
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
            with open(os.path.join(self.model_path, "logs", "Only_Test_Metrics.json"), "w") as file:
                json.dump(average_metrics, file, indent=4)
            json_formatted_str = json.dumps(average_metrics, indent=4)
            print("Test_Metrics: ", json_formatted_str.replace('"', ""))
