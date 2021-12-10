import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import os
from scipy.io import loadmat
from torchvision.datasets.folder import default_loader

DATA_ROOT = os.path.expanduser("data/cdfsl")


class TorchDataset(Dataset):
    """ parent class for torchvision datasets  """
    def __init__(self, ):
        super().__init__()

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index: int):
        image, label = self.dset[index]
        image = image.convert('RGB')
        return image, label


class miniImageNet(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        if mode is None or mode == '':
            mode = 'train'
        data = np.load(os.path.join(data_root,
                                    f'mini-imagenet-cache-{mode}.pkl'),
                       allow_pickle=True)
        self.image_data = data['image_data']
        self.class_dict = data['class_dict']
        self.classes = sorted(list(self.class_dict.keys()))
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imindex, label = self.samples[index]
        image = Image.fromarray(
            self.image_data[imindex].astype('uint8')).convert('RGB')
        label = np.long(label)
        return image, label

    def _make_dataset(self):
        instances = []
        for target_class in sorted(self.cls_to_idx.keys()):
            class_index = self.cls_to_idx[target_class]
            target_index = self.class_dict[target_class]
            instances.extend([(_ind, class_index) for _ind in target_index])
        return instances


class miniImageNettest(miniImageNet):
    def __init__(self, data_root, mode='train'):
        super().__init__(data_root, mode='test')


class tieredImageNet(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        partition = mode
        self.data_root = data_root
        self.partition = partition

        self.image_file_pattern = '%s_images.npz'
        self.label_file_pattern = '%s_labels.pkl'

        # modified code to load tieredImageNet
        image_file = os.path.join(self.data_root,
                                  self.image_file_pattern % partition)
        self.imgs = np.load(image_file)['images']
        label_file = os.path.join(self.data_root,
                                  self.label_file_pattern % partition)
        self.labels = self._load_labels(label_file)['labels']

    def __getitem__(self, item):
        img = Image.fromarray(np.asarray(self.imgs[item]).astype('uint8'))
        target = self.labels[item] - min(self.labels)
        return img, target

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _load_labels(file):
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data


class tieredImageNettest(tieredImageNet):
    def __init__(self, data_root, mode='train'):
        super().__init__(data_root, mode='test')


ChestX_path = os.path.expanduser("data/cdfsl/chest_xray")


class ChestX(Dataset):
    def __init__(self, data_root, mode='train', csv_path=ChestX_path+"/Data_Entry_2017.csv", \
        image_path = ChestX_path+"/images_resized/"):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax"
        ]

        self.labels_maps = {
            "Atelectasis": 0,
            "Cardiomegaly": 1,
            "Effusion": 2,
            "Infiltration": 3,
            "Mass": 4,
            "Nodule": 5,
            "Pneumothorax": 6
        }

        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name = []
        self.labels = []

        for name, label in zip(self.image_name_all, self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[
                    0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Open image
        # img_as_img = Image.open(self.img_path + single_image_name).resize(
        #     (256, 256)).convert('RGB')

        img_as_img = default_loader(self.img_path + single_image_name)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


ISIC_path = os.path.expanduser("data/cdfsl/ISIC")


class ISIC(Dataset):
    def __init__(self,data_root, mode='train', csv_path= ISIC_path + "/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv", \
        image_path =  ISIC_path + "/ISIC2018_Input_Resized/"):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path

        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.labels = np.asarray(self.data_info.iloc[:, 1:])
        self.labels = (self.labels != 0).argmax(axis=1)
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        img_as_img = default_loader(
            os.path.join(self.img_path, single_image_name + ".JPG"))

        single_image_label = self.labels[index]
        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


class CropDisease(datasets.ImageFolder):
    def __init__(self, data_root: str, mode='train'):
        # NOTE: load 'all' for cropdisease, as it would be used for few-shot
        # if mode == 'train':
        #     path = os.path.join(data_root, 'train')
        # elif mode == "test":
        #     path = os.path.join(data_root, 'test')
        # else:
        path = os.path.join(data_root, 'all')
        super().__init__(path)


class CUB(Dataset):
    def __init__(self, data_root: str, mode='train'):
        super().__init__()
        self.samples = self._make_dataset(data_root, mode)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        imfile, label = self.samples[index]
        image = default_loader(imfile)
        label = np.long(label)
        return image, label

    def _make_dataset(self, data_root, mode):
        img_root = os.path.join(data_root, 'images')

        def fn_read(path):
            _list = []
            with open(path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) > 0:
                        lab = int(line.split('.')[0]) - 1
                        imname = os.path.join(img_root, line)
                        assert os.path.exists(imname)
                        _list.append((imname, lab))
            return _list

        instances = []
        if mode == 'train':
            instances = fn_read(os.path.join(data_root, 'train.txt'))
        elif mode == 'test':
            instances = fn_read(os.path.join(data_root, 'test.txt'))
        else:
            instances = fn_read(os.path.join(data_root, 'train.txt'))
            instances.extend(fn_read(os.path.join(data_root, 'test.txt')))
        return instances


class ConcatProportionDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, len_mode='max', return_data_idx=False):
        super().__init__()
        self.datasets = datasets
        self.len_mode = len_mode
        self.return_data_idx = return_data_idx
        self.ptr = 0

    def __len__(self):
        if self.len_mode == 'max':
            return max([len(dat) for dat in self.datasets])
        else:
            return min([len(dat) for dat in self.datasets])

    def __getitem__(self, idx):
        # data_idx = int(np.random.choice(len(self.datasets)))
        data_idx = self.ptr
        self.ptr = (self.ptr + 1) % len(self.datasets)

        dataset = self.datasets[data_idx]
        data, label = self.get_data(dataset, idx)
        if self.return_data_idx:
            label = data_idx
        return data, label

    def get_data(self, dataset, idx):
        if self.len_mode == 'max' and idx >= len(dataset):
            idx = np.random.choice(len(dataset))
        elif self.len_mode != 'max':
            idx = np.random.choice(len(dataset))
        data = dataset[idx]
        return data
