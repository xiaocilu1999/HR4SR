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
    main()
    end = time.time()
