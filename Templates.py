import argparse

parser = argparse.ArgumentParser(description="Sequence Recommendation made by XiaoCiLu")
args = parser.parse_args()

print("-" * 80)
print('Please select the dataset you want to run: ')
print('\t1. ML-1M     2. Steam     3. Beauty')

x = input('select: ')
if x == '1' or x == '2' or x == '3':
    x = int(x)
else:
    print('Wrong dataset selection, default dataset ML-1M has been selected')
    x = 1

print("-" * 80)

if x == 1:
    args.max_step = 30
    args.dataset_name = "ml-1m"
    args.sequence_max_length = 40
    args.bert_dropout = 0.2
    args.bert_num_heads = 4
    args.mask_probable = 0.5
    args.slide_window_step = 10
    args.increment = 10
    args.theta = 0.2
    args.val_or_not = 50
    args.mask_ratio = 0.8
    args.bert_num_blocks = 2
elif x == 2:
    args.max_step = 70
    args.dataset_name = "steam"
    args.sequence_max_length = 30
    args.mask_probable = 0.2
    args.bert_num_heads = 4
    args.bert_dropout = 0.3
    args.slide_window_step = 3
    args.increment = 10
    args.theta = 0.2
    args.val_or_not = 80
    args.mask_ratio = 1
    args.bert_num_blocks = 2
elif x == 3:
    args.max_step = 10
    args.dataset_name = "Amazon_Beauty"
    args.sequence_max_length = 20
    args.bert_dropout = 0.3
    args.bert_num_heads = 4
    args.mask_probable = 0.3
    args.slide_window_step = 5
    args.increment = 10
    args.theta = 0.2
    args.val_or_not = 100
    args.mask_ratio = 0.7
    args.bert_num_blocks = 1

args.batch_size = 256
args.bert_hidden_size = 256

args.lr = 1e-3
args.gamma = 0.1
args.weight_decay = 1e-5
args.decay_step = 30

args.early_stop = 50
args.gpu_id = '0'
args.device = 'cuda'
args.num_epochs = 1999


args.min_rating = 0
args.min_item_interact = 5
args.min_user_interact = 5

args.best_metric = "NDCG@10"
args.metric_ks = [5, 10, 20]
