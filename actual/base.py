from torchvision.transforms import v2 as transformsv2
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from enum import Enum
from PIL import Image
import numpy as np
import pickle
import random
import torch
import os

dataset_part = Enum('dataset_part', [('TRAIN', 1), ('TEST', 2), ('EVAL', 3)])

def reset_seed(seed=42):
    """Completly resets the seed for reusability of runs."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class CustomCIFAR10L(Dataset):
    """Custom Dataset wrapper for CIFAR10 datasets with modifications (logits). Used for working with pytorch dataset within huggingface."""
    def __init__(self, root, dataset_part = dataset_part.TRAIN, transform=None):
        self.root = root
        self.dataset_part = dataset_part
        self.transform = transform
        self.data = []
        self.targets = []
        self.logits = []
        self.logits_aug = []

        if self.dataset_part == dataset_part.TRAIN:
             for i in range(1, 5):
                 data_file = os.path.join(self.root, 'cifar-10-batches-py', f'train_batch_{i}')
                 with open(data_file, 'rb') as fo:
                     dict = pickle.load(fo, encoding='bytes')
                     self.data.append(dict[b'data'])
                     self.targets.extend(dict[b'labels'])
                     self.logits.extend(dict[b'logits'])
                     self.logits_aug.extend(dict[b'logits_aug'])   

        elif self.dataset_part == dataset_part.TEST:
            data_file = os.path.join(self.root, 'cifar-10-batches-py', 'test')
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data.append(dict[b'data'])
                self.targets.extend(dict[b'labels'])
                self.logits.extend(dict[b'logits'])
        else:
            data_file = os.path.join(self.root, 'cifar-10-batches-py', 'eval')
            with open(data_file, "rb") as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data.append(dict[b'data'])
                self.targets.extend(dict[b'labels'])
                self.logits.extend(dict[b'logits'])
                
        self.data = np.concatenate(self.data, axis=0)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        label = self.targets[index]

        if self.transform:
            logit = self.logits[index] if len(self.transform.extra_repr()) < 300 else self.logits_aug[index]
            logit = torch.tensor(logit, dtype=torch.float)
            torch.manual_seed(index%10000)
            image = self.transform(image)            

        return {
            'pixel_values': image,
            'labels': label,
            'logits': logit,
        }
    
    def remove_entries(self, remove_list):
        self.data = np.delete(self.data, remove_list, axis=0)
        self.targets = np.delete(self.targets, remove_list, axis=0)
        self.logits = np.delete(self.logits, remove_list, axis=0)
        self.logits_aug = np.delete(self.logits_aug, remove_list, axis=0)
    
    @property
    def labels(self):
        return self.targets



class CustomCIFAR10(Dataset):
    """Custom Dataset wrapper for CIFAR10 datasets wihout any changes made. Used for working with pytorch dataset within huggingface."""

    def __init__(self, root, batch=None, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.train:
            self.data_file = os.path.join(self.root, 'cifar-10-batches-py', f'data_batch_{batch}')
        else:
            self.data_file = os.path.join(self.root, 'cifar-10-batches-py', 'test_batch')

        with open(self.data_file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.data = dict[b'data']
            self.labels = dict[b'labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        label = torch.as_tensor(self.labels[index])
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        if self.transform:            
            torch.manual_seed(index)
            image = self.transform(image)


        return  image.to(self.device), label.to(self.device)
    


def base_transforms():
    """Standard transformation for images within datasets (resize, normalize, convert)."""
    return transformsv2.Compose([
                transformsv2.ToImage(),
                transformsv2.ToDtype(torch.float32, scale=True),
                transformsv2.Resize((224, 224), antialias=True),
                transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

def aug_transforms():
    """Standard + Augmentation transformation for images within datasets (resize, normalize, convert) + (flips, rotation)."""
    return transformsv2.Compose([ 
                transformsv2.ToImage(),
                transformsv2.ToDtype(torch.float32, scale=True),
                transformsv2.Resize(size=(224, 224), antialias=True),
                transformsv2.RandomHorizontalFlip(),
                transformsv2.RandomVerticalFlip(),
                transformsv2.RandomRotation(15),
                transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

def unpickle(file):
    """Opening dataset files with pickle."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pickle_up(file, contents):
    """Saving dataset files with pickle."""
    with open(file, 'wb') as fo:
        pickle.dump(contents, fo, protocol=pickle.HIGHEST_PROTOCOL)


def generate_logits(dataloder, model):
    """Generates logits for given input."""
    logits_arr = []
    for batch in tqdm(dataloder):
        pixel_values, labels = batch
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
        logits_arr.append(logits.cpu().numpy())

    logits_arr_flat = []
    for tensor in logits_arr:
        logits_arr_flat.extend(tensor)
    return logits_arr_flat