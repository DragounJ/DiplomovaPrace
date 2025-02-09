from transformers import Trainer, TrainingArguments, MobileNetV2Config, MobileNetV2ForImageClassification, EarlyStoppingCallback
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm.notebook import tqdm
from enum import Enum
from PIL import Image
import torch.nn as nn
import numpy as np
import evaluate
import pickle
import random
import torch
import os

def get_dataset_part():
    """Returns ENUM of dataset_parts"""
    return Enum('dataset_part', [('TRAIN', 1), ('TEST', 2), ('EVAL', 3)])

dataset_part = get_dataset_part()

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

class CustomCIFAR100L(Dataset):
    """Custom Dataset wrapper for CIFAR100 datasets with modifications (logits). Used for working with pytorch dataset within huggingface."""
    def __init__(self, root, dataset_part = dataset_part.TRAIN, transform=None):
        self.root = root
        self.dataset_part = dataset_part
        self.transform = transform
        

        self.data = []
        self.targets = []
        self.logits = []
        self.logits_aug = []


        if self.dataset_part == dataset_part.TRAIN:
            data_file = os.path.join(self.root, 'cifar-100-python', 'train')
        elif self.dataset_part == dataset_part.TEST:
            data_file = os.path.join(self.root, 'cifar-100-python', 'test')
        else:
            data_file = os.path.join(self.root, 'cifar-100-python', 'eval')

        with open(data_file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.data.append(dict[b'data'])
            self.targets.extend(dict[b'fine_labels'])
            self.logits.extend(dict[b'logits'])  
            self.logits_aug.extend(dict[b'logits_aug'])   
            
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        label = self.targets[index]
        #logit = self.logits[index]
        
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        if self.transform:
            logit = self.logits[index] if len(self.transform.extra_repr()) < 300 else self.logits_aug[index]
            torch.manual_seed(index)
            image = self.transform(image)

        logit = torch.tensor(logit, dtype=torch.float)

        return {
            'pixel_values': image,
            'labels': label,
            'logits': logit
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
    
class CustomCIFAR100(Dataset):
    """Custom Dataset wrapper for CIFAR100 datasets wihout any changes made. Used for working with pytorch dataset within huggingface."""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.train:
            self.data_file = os.path.join(self.root, 'cifar-100-python', 'train')
        else:
            self.data_file = os.path.join(self.root, 'cifar-100-python', 'test')

        with open(self.data_file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.data = dict[b'data']
            self.labels = dict[b'fine_labels']


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



def check_acc(dataset):
    """Swift Acc checker for given dataset and its appended logits."""
    corr = []
    for val in tqdm(dataset):
        if torch.topk(val["logits"], k=1).indices.numpy()[0] == val["labels"]:  corr.append(True)
    
    return(f"Accuracy for given set is: {len(corr)/len(dataset)}")  


def remove_diff_pred_class(normal, aug):
    rem_ls = []
    for index, val in enumerate(aug):
        target_alt = torch.topk(val["logits"], k=1).indices.numpy()[0]
        target_act = torch.topk(normal[index]["logits"], k=1).indices.numpy()[0]
        if target_alt != target_act:
            rem_ls.append(index)
    aug.remove_entries(rem_ls)
    return aug


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    pred, labels = eval_pred
    predictions = np.argmax(pred, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro', zero_division = 0)
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro', zero_division = 0)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }


class Custom_training_args(TrainingArguments):
    def __init__(self, lambda_param, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.lambda_param = lambda_param
        self.temperature = temperature


def get_training_args(output_dir, logging_dir, remove_unused_columns=True, lr=5e-5, epochs=5, weight_decay=0, lambda_param=.5, temp=5):
    return (
        Custom_training_args(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr, #Defaultní hodnota 
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        seed = 42,  #Defaultní hodnota 
        metric_for_best_model="f1",
        load_best_model_at_end = True,
        fp16=True, 
        logging_dir=logging_dir,
        remove_unused_columns=remove_unused_columns,
        lambda_param = lambda_param, 
        temperature = temp
    ))


def get_random_init_mobilenet(num_labels):
    reset_seed(42)
    student_config = MobileNetV2Config()
    student_config.num_labels = num_labels
    return MobileNetV2ForImageClassification(student_config)


def get_mobilenet(num_labels):
    model_pretrained = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    in_features = model_pretrained.classifier.in_features

    model_pretrained.classifier = nn.Linear(in_features,num_labels) #Úprava klasifikační hlavy
    model_pretrained.num_labels = num_labels
    model_pretrained.config.num_labels = num_labels

    return model_pretrained


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

class ImageDistilTrainer(Trainer):
    def __init__(self, student_model=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = self.args.temperature
        self.lambda_param = self.args.lambda_param

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        logits = inputs.pop("logits")

        student_output = student(**inputs)

        soft_teacher = F.softmax(logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = student_output.loss

        loss = ((1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss)
        return (loss, student_output) if return_outputs else loss
    

