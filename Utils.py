import json
import os
import random
import re
import shutil

import numpy as np
import torch
from torch.backends import cudnn


def save_experiment_args(args):
    ROOT = "./Experimental_results/"
    experiment_path = (ROOT + args.dataset_name)
    dict_args = json.loads(json.dumps(vars(args)))
    if not os.path.exists(experiment_path):
        experiment_path = experiment_path + '/version_1'
        os.makedirs(experiment_path)
    else:
        file_list = os.listdir(experiment_path)
        number = get_index(file_list, experiment_path, dict_args)
        experiment_path = experiment_path + '/version_' + str(number)
        os.makedirs(experiment_path)
    print("Experimental results folder created: " + experiment_path)
    with open(os.path.join(experiment_path, "Experimental_Parameters.json"), "w") as outfile:
        json.dump(dict_args, outfile, indent=2)
    print('The hyperparameters have been saved')
    return experiment_path


def get_index(file_list, experiment_path, ori_data):
    all_num = []
    for name in file_list:
        number = re.findall("\\d+", name)[0]
        with open(os.path.join(experiment_path, name, "Experimental_Parameters.json")) as file:
            data = json.load(file)
        if ori_data == data:
            print('Folders with the same parameters: ' + name)
            shutil.rmtree(os.path.join(experiment_path, name))
            return number
        else:
            all_num.append(int(number))
    return max(all_num) + 1


def fix_random_seed_as(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def luck():
    print("-" * 100)
    print(" .d8888b.    .d88888b.    .d88888b.   8888888b.       888      888     888   .d8888b.   888    d8P  ")
    print("d88P  Y88b  d88P\" \"Y88b  d88P\" \"Y88b  888  \"Y88b      888      888     888  d88P  Y88b  888   d8P   ")
    print("888    888  888     888  888     888  888    888      888      888     888  888    888  888  d8P    ")
    print("888         888     888  888     888  888    888      888      888     888  888         888d88K     ")
    print("888  88888  888     888  888     888  888    888      888      888     888  888         8888888b    ")
    print("888    888  888     888  888     888  888    888      888      888     888  888    888  888  Y88b   ")
    print("Y88b  d88P  Y88b. .d88P  Y88b. .d88P  888  .d88P      888      Y88b. .d88P  Y88b  d88P  888   Y88b  ")
    print(
        "\"Y8888P88    \"Y88888P\"    \"Y88888P\"   8888888P\"       88888888  \"Y88888P\"    \"Y8888P\"   888    "
        "Y88b  ")
    print("-" * 100)


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.valid
            for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.average
            for name, meter in self.meters.items()
        }

    def sums(self, format_string="{}"):
        return {
            format_string.format(name): meter.sum for name, meter in self.meters.items()
        }

    def counts(self, format_string="{}"):
        return {
            format_string.format(name): meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter(object):

    def __init__(self):
        self.valid = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.valid = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, valid, n=1):
        self.valid = valid
        self.sum += valid
        self.count += n
        self.average = self.sum / self.count

    def __format__(self, format_):
        return "{self.valid:{format}} ({self.average:{format}})".format(
            self=self, format=format_
        )
