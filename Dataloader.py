import random

import torch
import torch.utils.data as data_utils


class Dataloader(object):
    def __init__(self, args, dataset):
        self.args = args
        self.batch_size = args.batch_size
        self.mask_probable = args.mask_probable
        self.slide_window_step = args.slide_window_step
        self.sequence_max_length = args.sequence_max_length
        self.mask_ratio = args.mask_ratio

        self.train_dataset = dataset["train"]
        self.valid_dataset = dataset['valid']
        self.test_dataset = dataset['test']
        self.user_count = len(dataset["user_map"])
        self.item_count = len(dataset["item_map"])
        self.MASK_TOKEN = self.item_count + 1

        self.train_slide_window = self.get_train_dataset_slide_window()
        self.dataset_analysis()

    def dataset_analysis(self):
        i, o = 0, 0
        for value in self.train_dataset.values():
            i = i + len(value)
        for value in self.train_slide_window.values():
            o = o + len(value)
        print("-" * 80)
        print("dataset average length: %.2f" % (i / len(self.train_dataset)))
        print("user count: %d" % self.user_count)
        print("-" * 80)
        print("sliding window average length: %.2f" % (o / len(self.train_slide_window)))
        print("pseudo user count: %d" % len(self.train_slide_window))
        print("-" * 80)

    def get_train_dataset_slide_window(self):
        real_user_count = 1
        train_slide_window = {}
        for user in range(1, self.user_count + 1):
            sequence = self.train_dataset[user]
            sequence_length = len(sequence)
            begin_index = list(range(sequence_length - self.sequence_max_length, 0, -self.slide_window_step))
            begin_index.append(0)
            for item in begin_index:
                temp = sequence[item: item + self.sequence_max_length]
                train_slide_window[real_user_count] = temp
                real_user_count += 1
        return train_slide_window

    def get_dataloaders(self):
        """获得三个迭代器"""
        train_loader = data_utils.DataLoader(
            dataset=TrainDataset(self.train_slide_window, self.MASK_TOKEN, self.mask_probable, self.mask_ratio,
                                 self.sequence_max_length),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        valid_loader = data_utils.DataLoader(
            dataset=ValidDataset(self.train_dataset, self.valid_dataset, self.sequence_max_length, self.MASK_TOKEN,
                                 self.item_count),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        test_loader = data_utils.DataLoader(
            dataset=TestDataset(self.train_dataset, self.valid_dataset, self.test_dataset, self.item_count,
                                self.MASK_TOKEN, self.sequence_max_length),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        data_loader = {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader,
        }

        return data_loader


class TrainDataset(data_utils.Dataset):
    def __init__(self, train_augmentation, MASK_TOKEN, mask_probable, mask_ratio, sequence_max_length):
        self.MASK_TOKEN = MASK_TOKEN
        self.mask_probable = mask_probable
        self.mask_ratio = mask_ratio
        self.train_dataset = train_augmentation
        self.sequence_max_length = sequence_max_length

        self.users = sorted(self.train_dataset.keys())
        self.random_numeral_generator = random.Random(0)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        sequence = self.train_dataset[user]

        mask_sequence, mask_sequence_label = self.get_mask_sequence(sequence)

        return mask_sequence, mask_sequence_label

    def get_mask_sequence(self, sequence):
        tokens = []
        labels = []

        for item in sequence:
            probable = self.random_numeral_generator.random()  # 如果只mask最后一个 第一轮 结果很差

            if probable < self.mask_probable:
                probable /= self.mask_probable

                if probable <= self.mask_ratio:
                    tokens.append(self.MASK_TOKEN)
                    labels.append(item)
                else:
                    tokens.append(item)
                    labels.append(item)
            else:
                tokens.append(item)
                labels.append(0)

            tokens = tokens[-self.sequence_max_length:]
            labels = labels[-self.sequence_max_length:]

            mask_length = self.sequence_max_length - len(tokens)

            tokens = [0] * mask_length + tokens
            labels = [0] * mask_length + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)


class ValidDataset(data_utils.Dataset):
    def __init__(self, train_dataset, valid_dataset, sequence_max_length, MASK_TOKEN, item_count):
        self.MASK_TOKEN = MASK_TOKEN
        self.item_count = item_count
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.sequence_max_length = sequence_max_length

        self.users = sorted(self.train_dataset.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        sequence = self.train_dataset[user]
        answer = self.valid_dataset[user]

        sequence = sequence + [self.MASK_TOKEN]
        sequence = sequence[-self.sequence_max_length:]

        padding_length = self.sequence_max_length - len(sequence)
        sequence = [0] * padding_length + sequence
        interacted = set(answer + sequence)  # 去掉重复交互
        candidates = answer + [
            x for x in range(1, self.item_count + 1) if x not in interacted
        ]
        candidates = candidates + [0] * (self.item_count - len(candidates))

        labels = [1] * len(answer) + [0] * (len(candidates) - 1)
        return (
            torch.LongTensor(sequence),
            torch.LongTensor(candidates),
            torch.LongTensor(labels),
        )


class TestDataset(data_utils.Dataset):
    def __init__(self, train_dataset, valid_dataset, test_dataset, item_count, MASK_TOKEN, sequence_max_length):
        self.item_count = item_count
        self.MASK_TOKEN = MASK_TOKEN
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.sequence_max_length = sequence_max_length

        self.users = sorted(self.train_dataset.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        sequence = self.train_dataset[user]
        valid = self.valid_dataset[user]
        answer = self.test_dataset[user]

        sequence = sequence + valid + [self.MASK_TOKEN]
        sequence = sequence[-self.sequence_max_length:]

        padding_length = self.sequence_max_length - len(sequence)
        sequence = [0] * padding_length + sequence
        interacted = set(answer + sequence)
        candidates = answer + [
            x for x in range(1, self.item_count + 1) if x not in interacted
        ]
        candidates = candidates + [0] * (self.item_count - len(candidates))

        labels = [1] * len(answer) + [0] * (len(candidates) - 1)

        return (
            torch.LongTensor(sequence),
            torch.LongTensor(candidates),
            torch.LongTensor(labels),
        )
