import os

from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, lablel_dir):
        self.root_dir = root_dir
        self.lablel_dir = lablel_dir
        self.path = os.path.join(self.root_dir, self.lablel_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        image_name = self.img_path[index]
        img_item_path = os.path.join(self.path, image_name)
        img = Image.open(img_item_path)
        lebal = self.lablel_dir
        return img, lebal

    def __len__(self):
        return len(self.img_path)


root_dir = "../dataset/train"
ant_label_dir = "ants"
ants_Dataset = MyDataset(root_dir, ant_label_dir)
bee_label_dir = "bees"
bees_Dataset = MyDataset(root_dir, bee_label_dir)

train_dataset = ants_Dataset + bees_Dataset
