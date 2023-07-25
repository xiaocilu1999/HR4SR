import os
import time

from Dataloader import Dataloader
from Dataset import Dataset
from DiffusionBERT import DiffusionBERT
from Inference import Inference
from Templates import args
from Trainer import Trainer
from Utils import save_experiment_args, luck


def main():

    luck()

    if args.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        experiment_path = save_experiment_args(args)
        dataset, num_item = Dataset(args).make_and_load_pkl()
        data_loader = Dataloader(args, dataset).get_dataloaders()
        model = DiffusionBERT(args, num_item)
        Trainer(args, data_loader, model, experiment_path).train()
    elif args.mode == 'inference':
        model_path = r'.\Experimental_results\ml-1m'
        dataset, num_item = Dataset(args).make_and_load_pkl()
        data_loader = Dataloader(args, dataset).get_dataloaders()
        model = DiffusionBERT(args, num_item)
        Inference(args, model, model_path, data_loader).inference()
    else:
        print("look look you args! ")


if __name__ == '__main__':
    args.mode = 'train'  # 选择模式
    start = time.time()
    for args.mask_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        args.push_title = 'args.mask_ratio = ' + str(args.mask_ratio)
        main()
    end = time.time()
