import os
import pickle

import pandas as pd
from tqdm import tqdm


class Dataset(object):
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.min_rating = args.min_rating
        self.min_user_interact = args.min_user_interact
        self.min_item_interact = args.min_item_interact
        assert self.min_user_interact >= 2

    def make_and_load_pkl(self):
        ROOT = "./Data/Preprocess/"
        NAME = self.dataset_name + ".pkl"
        PATH = "./Data/" + self.dataset_name + "/" + self.dataset_name + ".inter"

        if os.path.isfile(ROOT + NAME):
            print("Already preprocessed. Skip preprocessing!")
            with open(ROOT + NAME, "rb") as file:
                dataset = pickle.load(file)
        else:
            if not os.path.exists(ROOT):
                os.makedirs(ROOT)
            data_frame = self.load_data_frame(PATH)
            if self.min_rating > 0:
                data_frame = self.make_implicit(data_frame)

            if self.min_user_interact or self.min_item_interact > 0:
                data_frame = self.filter_triplets(data_frame)

            data_frame, user_map, item_map = self.make_map(data_frame)

            train, valid, test = self.split_data_frame(data_frame, user_map)

            dataset = {
                "train": train,
                "valid": valid,
                "test": test,
                "user_map": user_map,
                "item_map": item_map,
            }

            with open(ROOT + NAME, "wb") as file:
                pickle.dump(dataset, file)

        num_item = len(dataset['item_map'])
        num_user = len(dataset['user_map'])
        print('-'*80)
        print('user_numbers: ', num_user)
        print('item_numbers: ', num_item)

        return dataset, num_item

    def load_data_frame(self, PATH):
        print("Loading " + self.dataset_name + " ......")
        data_frame = pd.read_csv(PATH, sep="	", engine="python")
        data_frame.columns = ["user_id", "item_id", "rating", "timestamp"]
        return data_frame

    def make_implicit(self, data_frame):
        print("Making implicit ......")
        data_frame = data_frame[data_frame["rating"] >= self.min_rating]
        return data_frame

    def filter_triplets(self, data_frame):
        print("Filtering triplets ......")
        user_count = [0, 0]
        item_count = [0, 0]
        while True:
            user_count[0] = len(set(data_frame["user_id"]))
            item_count[0] = len(set(data_frame["item_id"]))
            if self.min_item_interact > 0:
                item_sizes = data_frame.groupby("item_id").size()
                good_items = item_sizes.index[item_sizes >= self.min_item_interact]
                data_frame = data_frame[data_frame["item_id"].isin(good_items)]
            if self.min_user_interact > 0:
                user_sizes = data_frame.groupby("user_id").size()
                good_users = user_sizes.index[user_sizes >= self.min_user_interact]
                data_frame = data_frame[data_frame["user_id"].isin(good_users)]
            user_count[1] = len(set(data_frame["user_id"]))
            item_count[1] = len(set(data_frame["item_id"]))
            if user_count[0] == user_count[1] and item_count[0] == item_count[1]:
                break
        return data_frame

    @staticmethod
    def make_map(data_frame):
        print("Making mapp ......")
        user_map = {user: (i + 1) for i, user in enumerate(sorted(set(data_frame["user_id"])))}
        item_map = {item: (i + 1) for i, item in enumerate(sorted(set(data_frame["item_id"])))}
        data_frame["user_id"] = data_frame["user_id"].map(user_map)
        data_frame["item_id"] = data_frame["item_id"].map(item_map)

        return data_frame, user_map, item_map

    @staticmethod
    def split_data_frame(data_frame, user_map):
        tqdm.pandas(desc="Splitting data frame:")
        user_group = data_frame.groupby("user_id")

        user2items = user_group.progress_apply(lambda x: list(x.sort_values(by="timestamp")["item_id"]))

        train, valid, test = {}, {}, {}

        for user in range(1, len(user_map) + 1):
            items = user2items[user]
            train[user], valid[user], test[user] = items[:-2], items[-2:-1], items[-1:]

        return train, valid, test
