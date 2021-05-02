from torch.utils.data import Dataset, DataLoader
import random
import csv
import os

from torchvision import transforms

from PIL import Image

def get_dataloader(data_dir, batch_size=32, num_workers=8, train_shuffle=True, normalize_image=True):
    pass

    class EnricoDataset(Dataset):
        def __init__(self, data_dir, mode="train", img_dim=224, random_seed=42, train_split=0.7, val_split=0.15, test_split=0.15, normalize_image=True):
            super(EnricoDataset, self).__init__()
            self.img_dim = img_dim
            csv_file = os.path.join(data_dir, "design_topics.csv")
            self.img_dir = os.path.join(data_dir, "screenshots")
            self.hierarchy_dir = os.path.join(data_dir, "hierarchies")
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                example_list = list(reader)

            self.example_list = example_list

            keys = list(range(len(example_list)))
            # shuffle and create splits
            random.Random(random_seed).shuffle(keys)
            
            if mode == "train":
                # train split is at the front
                start_index = 0
                stop_index = int(len(example_list) * train_split)
            elif mode == "val":
                # val split is in the middle
                start_index = int(len(example_list) * train_split)
                stop_index = int(len(example_list) * (train_split + val_split))
            elif mode == "test":
                # test split is at the end
                start_index = int(len(example_list) * (train_split + val_split))
                stop_index = len(example_list)

            # only keep examples in the current split
            keys = keys[start_index:stop_index]
            self.keys = keys

            img_transforms = [
                transforms.Resize((img_dim, img_dim)),
                transforms.ToTensor()
            ]
            if normalize_image:
                img_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

            # pytorch image transforms
            self.img_transforms = transforms.Compose(img_transforms)

            # make maps
            topics = set()
            for e in example_list:
                topics.add(e['topic'])
            topics = sorted(list(topics))

            idx2Topic = {}
            topic2Idx = {}

            for i in range(len(topics)):
                idx2Topic[i] = topics[i]
                topic2Idx[topics[i]] = i

            self.idx2Topic = idx2Topic
            self.topic2Idx = topic2Idx

        def __len__(self):
            return len(keys)

        def __getitem__(self, idx):
            example = self.example_list[self.keys[idx]]
            screenId = example['screen_id']
            screenImg = Image.open(os.path.join(self.img_dir, screenId + ".jpg"))
            screenImg = self.img_transforms(screenImg)

            screenLabel = self.topic2Idx[example['topic']]
            # return a list where each index is a modality
            

    ds_train = EnricoDataset(data_dir, mode="train")
    ds_val = EnricoDataset(data_dir, mode="val")
    ds_test = EnricoDataset(data_dir, mode="test")

    dl_train = DataLoader(ds_train, shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    dl_val = DataLoader(ds_val, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    dl_test = DataLoader(ds_test, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return dl_train, dl_val, dl_test