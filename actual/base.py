from transformers import Trainer, TrainingArguments, MobileNetV2Config, MobileNetV2ForImageClassification, BasicTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import Dataset
from torch.utils import benchmark
import torch.nn.functional as F
from tqdm.notebook import tqdm
from enum import Enum
from PIL import Image
import torch.nn as nn
import pandas as pd
import numpy as np
import evaluate
import pickle
import random
import torch
import nltk
import os



nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def get_dataset_part():
    """Returns ENUM of dataset_parts"""
    #ENUM pro potřeby CIFAR dataset WRAPPERů.
    return Enum('dataset_part', [('TRAIN', 1), ('TEST', 2), ('EVAL', 3)])

dataset_part = get_dataset_part()

def reset_seed(seed=42):
    """Completly resets the seed for reusability of runs."""
    #Nastavení všech možných seedů pro zajištění reproducibility běhu.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def base_transforms():
    """Standard transformation for images within datasets (resize, normalize, convert)."""
    #Základní transformace obrázku pro dataset CIFAR10/100.
    #Převod datového typu, změna velikosti a normalizace hodnot.
    return transformsv2.Compose([
                transformsv2.ToImage(),
                transformsv2.ToDtype(torch.float32, scale=True),
                transformsv2.Resize((224, 224), antialias=True),
                transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

def aug_transforms():
    """Standard + Augmentation transformation for images within datasets (resize, normalize, convert) + (flips, rotation)."""
    #Rozšíření základních transformací o augmentační.
    #Přidání rotace a převracení os.
    return transformsv2.Compose([ 
                transformsv2.ToImage(),
                transformsv2.ToDtype(torch.float32, scale=True),
                transformsv2.Resize(size=(224, 224), antialias=True),
                transformsv2.RandomHorizontalFlip(),
                transformsv2.RandomVerticalFlip(),
                transformsv2.RandomRotation(15),
                transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

class CustomCIFAR10L(Dataset):
    """Custom Dataset wrapper for CIFAR10 datasets with modifications (logits). Used for working with pytorch dataset within huggingface."""
    #Wrapper pro dataset CIFAR10, který umožňuje pracovat se staženými soubory a rozšiřovat je o nová data, primárně logity.
    def __init__(self, root, dataset_part = dataset_part.TRAIN, transform=None, device=torch.device("cpu")):
        self.root = root
        self.dataset_part = dataset_part
        self.transform = transform
        self.device = device

        self.data = []
        self.targets = []
        self.logits = []
        self.logits_aug = []

        #Zpracování všech souborů tvořící trénovací část.
        if self.dataset_part == dataset_part.TRAIN:
             for i in range(1, 5):
                 data_file = os.path.join(self.root, 'cifar-10-batches-py', f'train_batch_{i}')
                 with open(data_file, 'rb') as fo:
                     #Uložení dat do objektu wrapperu.
                     dict = pickle.load(fo, encoding='bytes')
                     self.data.append(dict[b'data'])
                     self.targets.extend(dict[b'labels'])
                     self.logits.extend(dict[b'logits'])
                     self.logits_aug.extend(dict[b'logits_aug'])   
        else:
            #Načtení testovací části datasetu.
            if self.dataset_part == dataset_part.TEST:
                data_file = os.path.join(self.root, 'cifar-10-batches-py', 'test')
            
            else:
                #Načtení validační části datasetu.
                data_file = os.path.join(self.root, 'cifar-10-batches-py', 'eval')
            with open(data_file, "rb") as fo:
                #Uložení dat do objektu wrapperu.
                dict = pickle.load(fo, encoding='bytes')
                self.data.append(dict[b'data'])
                self.targets.extend(dict[b'labels'])
                self.logits.extend(dict[b'logits'])
                
        self.data = np.concatenate(self.data, axis=0)
        
    #K získání počtu záznamů.
    def __len__(self):
        return len(self.data)

    #K získání jednoho záznamu na základě indexu.
    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        label = torch.as_tensor(self.labels[index])

        #Přiřazení správného druhu předpočítaných logitů na základě transformací (logity pro augmentované obrázky / logity pro původní obrázky).
        if self.transform:
            logit = self.logits[index] if len(self.transform.extra_repr()) < 300 else self.logits_aug[index]
            logit = torch.tensor(logit, dtype=torch.float)
            torch.manual_seed(index%10000)
            image = self.transform(image)            

        return {
            'pixel_values': image.to(self.device),
            'labels': label.to(self.device),
            'logits': logit.to(self.device),
        }
    
    #K odstranění nevyhovujícíh záznamů. Využíváno při filtraci augmentovaných dat.
    def remove_entries(self, remove_list):
        self.data = np.delete(self.data, remove_list, axis=0)
        self.targets = np.delete(self.targets, remove_list, axis=0)
        self.logits = np.delete(self.logits, remove_list, axis=0)
        self.logits_aug = np.delete(self.logits_aug, remove_list, axis=0)
    
    #K získání anotací.
    @property
    def labels(self):
        return self.targets

class CustomCIFAR100L(Dataset):
    """Custom Dataset wrapper for CIFAR100 datasets with modifications (logits). Used for working with pytorch dataset within huggingface."""
    #Wrapper pro dataset CIFAR100, který umožňuje pracovat se staženými soubory a rozšiřovat je o nová data, primárně logity.
    def __init__(self, root, dataset_part = dataset_part.TRAIN, transform=None, device=torch.device("cpu")):
        self.root = root
        self.dataset_part = dataset_part
        self.transform = transform
        self.device = device

        self.data = []
        self.targets = []
        self.logits = []
        self.logits_aug = []

        #Načtení trénovací části datasetu.
        if self.dataset_part == dataset_part.TRAIN:
            data_file = os.path.join(self.root, 'cifar-100-python', 'train')  
            with open(data_file, 'rb') as fo:
                #Uložení logitů pro augmentované obrázky do objektu wrapperu.
                dict = pickle.load(fo, encoding='bytes')
                self.logits_aug.extend(dict[b'logits_aug'])

        #Načtení testovací části datasetu.        
        elif self.dataset_part == dataset_part.TEST:
            data_file = os.path.join(self.root, 'cifar-100-python', 'test') 
        else:
            #Načtení validační části datasetu.
            data_file = os.path.join(self.root, 'cifar-100-python', 'eval')

        with open(data_file, 'rb') as fo:
            #Uložení dat do objektu wrapperu.
            dict = pickle.load(fo, encoding='bytes')
            self.data.append(dict[b'data'])
            self.targets.extend(dict[b'fine_labels'])
            self.logits.extend(dict[b'logits'])  
            
        self.data = np.concatenate(self.data, axis=0)

    #K získání počtu záznamů.
    def __len__(self):
        return len(self.data)
    
    #K získání jednoho záznamu na základě indexu.
    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        label = torch.as_tensor(self.labels[index])
        
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        #Přiřazení správného druhu předpočítaných logitů na základě transformací (logity pro augmentované obrázky / logity pro původní obrázky).
        if self.transform:
            logit = self.logits[index] if len(self.transform.extra_repr()) < 300 else self.logits_aug[index]
            torch.manual_seed(index)
            image = self.transform(image)
        else:
            logit = self.logits[index]

        logit = torch.tensor(logit, dtype=torch.float)

        return {
            'pixel_values': image.to(self.device),
            'labels': label.to(self.device),
            'logits': logit.to(self.device),
        }
    
    #K odstranění nevyhovujícíh záznamů. Využíváno při filtraci augmentovaných dat.
    def remove_entries(self, remove_list):
        self.data = np.delete(self.data, remove_list, axis=0)
        self.targets = np.delete(self.targets, remove_list, axis=0)
        self.logits = np.delete(self.logits, remove_list, axis=0)
        self.logits_aug = np.delete(self.logits_aug, remove_list, axis=0)

    #K získání anotací.   
    @property
    def labels(self):
        return self.targets

class CustomCIFAR10(Dataset):
    """Custom Dataset wrapper for CIFAR10 datasets wihout any changes made. Used for working with pytorch dataset within huggingface."""
    #Wrapper pro dataset CIFAR10, který umožňuje pracovat s původními staženými soubory.
    def __init__(self, root, batch=None, train=True, transform=None, device=torch.device("cpu")):
        self.root = root
        self.train = train
        self.transform = transform
        self.device = device

        #Načtení trénovací části datasetu z daného souboru. 
        if self.train:
            self.data_file = os.path.join(self.root, 'cifar-10-batches-py', f'data_batch_{batch}')
        #Načtení testovací části datasetu. 
        else:
            self.data_file = os.path.join(self.root, 'cifar-10-batches-py', 'test_batch')
        #Uložení dat do objektu wrapperu.
        with open(self.data_file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.data = dict[b'data']
            self.labels = dict[b'labels']

    #K získání počtu záznamů.
    def __len__(self):
        return len(self.data)
    
    #K získání jednoho záznamu na základě indexu.
    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        label = torch.as_tensor(self.labels[index])
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        if self.transform:      
            #Pro zajištění shodných augmentací.      
            torch.manual_seed(index)
            image = self.transform(image)


        return  {
            "pixel_values": image.to(self.device), 
            "labels": label.to(self.device)
            }
    
class CustomCIFAR100(Dataset):
    """Custom Dataset wrapper for CIFAR100 datasets wihout any changes made. Used for working with pytorch dataset within huggingface."""
    #Wrapper pro dataset CIFAR100, který umožňuje pracovat s původními staženými soubory.
    def __init__(self, root, train=True, transform=base_transforms(), device=torch.device("cpu")):
        self.root = root
        self.train = train
        self.transform = transform
        self.device = device

        #Načtení trénovací části datasetu. 
        if self.train:
            self.data_file = os.path.join(self.root, 'cifar-100-python', 'train')
        #Načtení testovací části datasetu. 
        else:
            self.data_file = os.path.join(self.root, 'cifar-100-python', 'test')
        #Uložení dat do objektu wrapperu.
        with open(self.data_file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            self.data = dict[b'data']
            self.labels = dict[b'fine_labels']

    #K získání počtu záznamů.
    def __len__(self):
        return len(self.data)
    
    #K získání jednoho záznamu na základě indexu.
    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        label = torch.as_tensor(self.labels[index])
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        if self.transform:
            #Pro zajištění shodných augmentací.             
            torch.manual_seed(index)
            image = self.transform(image)

        return  {
            "pixel_values": image.to(self.device), 
            "labels": label.to(self.device)
            }
        

def unpickle(file):
    """Opening dataset files with pickle."""
    #Načítání stažených souborů skrze pickle.
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pickle_up(file, contents):
    """Saving dataset files with pickle."""
    #Ukládání stažených souborů skrze pickle.
    with open(file, 'wb') as fo:
        pickle.dump(contents, fo, protocol=pickle.HIGHEST_PROTOCOL)


def generate_logits(dataloader, model):
    """Generates logits for given input."""
    #Pro daná data provede inferenci skrze daný model. Využíváno pro předpočítání logitů učitelem.
    logits_arr = []
    for batch in tqdm(dataloader, desc="Generating logits for given dataset: "):
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs["logits"]
            
        logits_arr.extend(logits.cpu().numpy())
    return logits_arr



def check_acc(dataset, desc="Accuracy for given set is: "):
    """Swift Acc checker for given dataset and its appended logits."""
    #Získání accuracy pro daný dataset na základě uložených logitů.
    corr = []
    for val in tqdm(dataset, desc="Calculating accuracy based on the saved logits: "): 
        if torch.topk(val["logits"], k=1).indices.numpy()[0] == val["labels"]:  corr.append(True)
    
    return(f"{desc} {len(corr)/len(dataset)}")  


def remove_diff_pred_class(normal, aug, pytorch_dataset=True):
    """Removes those entries from aug, that do not have the same biggest logit as normal."""
    #Odstranění těch záznamů, které mají jinou predikci než původní dataset.
    rem_ls = []
    for index, val in tqdm(enumerate(aug), total=len(aug), desc="Removing entries from augmented dataset that are different from the base one - based on saved logits: "):
        #Predikce učitele nad původním datasetem.
        target_alt = torch.topk(val["logits"], k=1).indices.numpy()[0]
        #Predikce učitele nad augmentovaným datasetem.
        target_act = torch.topk(normal[index]["logits"], k=1).indices.numpy()[0]
        #Pokud se liší, dojde k odstranění augmentovaného záznamu.
        if target_alt != target_act:
            rem_ls.append(index)
    if pytorch_dataset:
        aug.remove_entries(rem_ls)
    else:
        indices_to_keep = [i for i in range(len(aug)) if i not in rem_ls]
        aug = aug.select(indices_to_keep)
    return aug


def compute_metrics(eval_pred):
    """Computes metrics for HuggingFace trainer."""
    #Vypočítání metrik pro HuggingFace trainer.
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    pred, labels = eval_pred
    predictions = np.argmax(pred, axis=1)
    
    #Macro průměrování pro vícetřídní klasifikaci.
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
    """Custom wrapper of training args for distillation."""
    #Upravené trénovací parametry o nové parametry pro distilaci.
    def __init__(self, lambda_param, temperature, alpha_param, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.lambda_param = lambda_param
        self.temperature = temperature
        self.alpha_param = alpha_param        

def get_training_args(output_dir, logging_dir, remove_unused_columns=True, lr=5e-5, epochs=5, weight_decay=0, adam_beta1 = .9, lambda_param=.5, temp=5, batch_size=128, num_workers=4, alpha_param = .5, warmup_steps=0):
    """Returns training args that can be adjusted."""
    #Vrátí trénovací parametry, které lze upravit dle potřeby.
    #V základu obsahují výchozí hodnoty.
    return (
        Custom_training_args(
        output_dir=output_dir,
        eval_strategy="epoch",
        adam_beta1 = adam_beta1,
        warmup_steps = warmup_steps,
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr, 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        seed = 42,  
        metric_for_best_model="f1",
        load_best_model_at_end = True,
        fp16=True, 
        logging_dir=logging_dir,
        remove_unused_columns=remove_unused_columns,
        lambda_param = lambda_param, 
        alpha_param = alpha_param, 
        temperature = temp,
        dataloader_num_workers=num_workers,

    ))

def get_random_init_mobilenet(num_labels):
    """Returns randomly initialized MobileNetV2."""
    #Vrátí náhodně inicializovaný MobileNetV2.
    reset_seed(42)
    student_config = MobileNetV2Config()
    student_config.num_labels = num_labels
    return MobileNetV2ForImageClassification(student_config)


def get_mobilenet(num_labels):
    """Returns initialized MobileNetV2."""
    #Vrátí předtrénovaný MobileNetV2.
    model_pretrained = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    in_features = model_pretrained.classifier.in_features
    #Úprava klasifikační hlavy
    model_pretrained.classifier = nn.Linear(in_features,num_labels) 
    model_pretrained.num_labels = num_labels
    model_pretrained.config.num_labels = num_labels

    return model_pretrained


def freeze_model(model):
    """Freezes all params apart from classification head."""
    #Zmražení všech parametrů kromě klasifikační hlavy. Pro potřeby experimentů nad MobileNetV2.
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

class DistilTrainer(Trainer):
    """Distilation trainer, computes loss with logits from teacher in mind. Logits are precomputed."""
    #Upravený trenér pro trénink s destilací znalostí.
    def __init__(self, student_model=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        #Získání studenta a konfigurace destilace s nastavením destilační ztrátové funkce.
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = self.args.temperature
        self.lambda_param = self.args.lambda_param

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        #Získání předpočítaných logitů.
        logits = inputs.pop("logits")
        #Získání predikcí studenta.
        student_output = student(**inputs)
        #Výpočet ztráty na základě předpočítaných logitů.
        soft_teacher = F.softmax(logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output['logits'] / self.temperature, dim=-1)

        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = student_output["loss"]
        #Výpočet konečné ztráty kombinací ztráty studenta a destilace.
        loss = ((1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss)
        return (loss, student_output) if return_outputs else loss

class DistilTrainerInfer(Trainer):
    """Distilation trainer, computes loss with logits from teacher in mind. Logits are NOT precomputed."""
    #Upravený trenér pro trénink s destilací znalostí, výpočet logitů učitele probíhá za běhu tréninku.
    def __init__(self, student_model=None, teacher_model=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        #Získání studenta, učitele a konfigurace destilace s nastavením ztrátové funkce.
        self.student = student_model
        self.teacher = teacher_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = self.args.temperature
        self.lambda_param = self.args.lambda_param

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        #Zahození předpočítaných logitů z datového vstupu.
        _ = inputs.pop("logits")
        #Získání predikcí studenta.
        student_output = student(**inputs)
        #Získání predikcí učitele.
        with torch.no_grad():
            teacher_output = self.teacher(**inputs)
        #Výpočet ztráty destilace. 
        soft_teacher = F.softmax(teacher_output['logits'] / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output['logits'] / self.temperature, dim=-1)

        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = student_output["loss"]
        #Výpočet konečné ztráty kombinací ztráty studenta a destilace.
        loss = ((1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss)
        return (loss, student_output) if return_outputs else loss
    

class DistilTrainerInferText(Trainer):
    """Distilation trainer, computes loss with logits from teacher in mind. Logits are NOT precomputed. For text classification."""
    #Upravený trenér pro trénink s destilací znalostí, výpočet logitů učitele probíhá za běhu tréninku. Určen pro textovou klasifikaci (práce s různými indexy slov pro studenta a učitele).
    def __init__(self, student_model=None, teacher_model=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        #Získání studenta, učitele a konfigurace destilace s nastavením ztrátové funkce.
        self.student = student_model
        self.teacher = teacher_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = self.args.temperature
        self.lambda_param = self.args.lambda_param

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        #Zahození předpočítaných logitů z datového vstupu.
        _ = inputs.pop("logits")
        #Získání ID a attention pro učitele.
        teacher_ids = inputs.pop("teacher_ids")
        teacher_attention = inputs.pop("teacher_attention")

        #Získání predikcí studenta.
        student_output = student(**inputs)
        #Získání predikcí učitele.
        with torch.no_grad():
            teacher_output = self.teacher(teacher_ids, attention_mask=teacher_attention, labels=inputs["labels"])
        #Výpočet ztráty destilace. 
        soft_teacher = F.softmax(teacher_output['logits'] / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output['logits'] / self.temperature, dim=-1)
        
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = student_output["loss"]
        #Výpočet konečné ztráty kombinací ztráty studenta a destilace.
        loss = ((1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss)
        return (loss, student_output) if return_outputs else loss
    
class DistilTrainerInner(Trainer):
    """Distilation trainer, computes loss with logits and inner states from teacher in mind. Logits are precomputed."""
    #Upravený trenér pro trénink s destilací znalostí, výpočet logitů učitele probíhá za běhu tréninku. Určeno pro destilaci vnitřních stavů, předpokládá modely BERT a BERT TINY.
    def __init__(self, student_model=None, teacher_model = None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        #Získání studenta, učitele a konfigurace destilace s nastavením ztrátových funkcí.
        self.student = student_model
        self.teacher = teacher_model
        self.layer_loss_function = nn.MSELoss()
        self.logit_loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = self.args.temperature
        self.lambda_param = self.args.lambda_param
        self.alpha_param = self.args.alpha_param
        #Vytvoření lineární vrstvy pro projekci skrytých stavů učitele na studenta.
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.student_to_teacher = nn.Linear(128, 768).to(self.device)
        self.model_parameters = list(self.model.parameters()) + list(self.student_to_teacher.parameters())

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        #Získání předpočítaných logitů.
        logits = inputs.pop("logits")
        #Získání predikcí studenta včetně vnitřních stavů.
        student_output = student(**inputs, output_hidden_states=True)
        student_target_loss = student_output["loss"]
        #Získání predikcí učitele včetně vnitřních stavů.
        with torch.no_grad():
            teacher_output = self.teacher(**inputs, output_hidden_states=True)


        #Získání skrytých stavů učitele a studenta.
        teacher_hidden_states = teacher_output.hidden_states
        student_hidden_states = student_output.hidden_states

        #Získání skrytých stavů učitele a studenta pro dané vrstvy.
        teacher_l6 = teacher_hidden_states[6] / self.temperature
        teacher_l12 = teacher_hidden_states[12] / self.temperature
        student_l1 = student_hidden_states[1]
        student_l2 = student_hidden_states[2] 

        student_l1_projection = self.student_to_teacher(student_l1) / self.temperature
        student_l2_projection = self.student_to_teacher(student_l2) / self.temperature
        ##Výpočet ztráty destilace vnitřních stavů skrze projekci.
        layer_distillation_loss = (
            self.layer_loss_function(student_l1_projection, teacher_l6) +
            self.layer_loss_function(student_l2_projection, teacher_l12)
        )

        
        #Výpočet ztráty destilace logitů. 
        soft_teacher = F.softmax(logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output['logits'] / self.temperature, dim=-1)

        logit_distillation_loss = self.logit_loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        logit_label_loss = ((1. - self.lambda_param) * student_target_loss + self.lambda_param * logit_distillation_loss)

        #Výpočet konečné ztráty kombinací ztráty studenta, destilace logitů a destilace vnitřních stavů.
        loss = (1 - self.alpha_param) * logit_label_loss + self.alpha_param * layer_distillation_loss

        
        return (loss, student_output) if return_outputs else loss
    #Pro získání skóre na daném modelu, bylo třeba upravit předávaná data.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        logits = inputs.pop("logits")
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)
            loss = outputs.loss if "loss" in outputs else None
            logits = outputs.logits
        labels = inputs.get("labels")
        return loss, logits, labels
        
def prepare_dataset_teacher(dataset, tokenizer):
    """Prepares dataset for teacher model. Tokenizes dataset, adds padding and retrieves IDs and Attention Mask."""
    #Připraví dataset pro učitele. Tokenizuje dataset, přidá padding a získá ID a Attention Mask. Vrací pouze nové sloupce.
    dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', return_tensors="pt", max_length=60), batched=True, desc="Tokenizing the provided dataset")
    return (dataset["input_ids"], dataset["attention_mask"])


def count_parameters(model):
    """Counts number of trainable parameters in the model and its size."""
    #Získání počtu trénovatelných parametrů v modelu a jeho velikosti.
    table_header = [["Modules", "Parameters"]]
    table = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.append([name, params])
        total_params += params

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB.'.format(size_all_mb))
    table = pd.DataFrame(table, None, table_header)
    table.reset_index(drop=True, inplace=True)
    print(f"Total Trainable Params: {total_params}.")
    return table
    
def prepare_dataset(dataset, tokenizer):
    """Prepares dataset for teacher model. Tokenizes dataset, adds padding and retrieves IDs and Attention Mask. Returns whole changed dataset."""
    #Připraví dataset pro učitele. Tokenizuje dataset, přidá padding a získá ID a Attention Mask. Vrací celý upravený dataset.
    dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', return_tensors="pt", max_length=60), batched=True, desc="Tokenizing the provided dataset")
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', "attention_mask"], device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return dataset


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for text classification."""
    #Bidirekcionální LSTM klasifikátor pro textovou klasifikaci.
    def __init__(self, embedding_dim, hidden_dim, fc_dim, output_dim, embedding_matrix, freeze_embed = True):
        super(BiLSTMClassifier, self).__init__()
        #Vložení embedding matice pro daný dataset. Konfigurace jejího zmražení pro uchování reprezentací. 
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embed)
        #Konfigurace jednotlivých vrstev.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)  
        self.dropout = nn.Dropout(.2)
        self.fc2 = nn.Linear(fc_dim, output_dim)

    #Výpčet dopředného kroku skrze jednotlivé vrstvy.
    def forward(self, input_ids, labels=None):
        #Získání vektoru pro daný token a jeho přesun do BiLSTM vrstvy.
        embedded = self.embedding(input_ids)  
        _, (h_n, _) = self.lstm(embedded)
        #Last forward hidden state
        h_forward = h_n[-2, :, :]  
        #Last backward hidden state
        h_backward = h_n[-1, :, :]  
        #Spojení obou skrytých stavů do jednoho vektoru.
        out_cat = torch.cat((h_forward, h_backward), dim=1)
        #Provedení dropout vrstvy a klasifikační vrstvy.
        fc1_out = F.relu(self.fc1(out_cat))
        dropped = self.dropout(fc1_out)
        logits = self.fc2(dropped)
        ##Pokud je zadán label, dojde k výpočtu ztráty.
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss() 
            loss = loss_fn(logits, labels)
            return {"loss" : loss, "logits" : logits}
        return {"loss" : None, "logits": logits}
       

def get_pos_tag_word_map(sentences, tokenizer=BasicTokenizer(do_lower_case=True)):
    """Generates a map of POS tags to words from the given sentences."""
    #Generuje mapu POS tagů k jednotlivým tokenům z daných vět.
    pos_tag_word_map = {}
    for sentence in sentences:
        #Tokenizuje vstup a pro každý token získá POS tag.
        for token, pos_tag in nltk.pos_tag(tokenizer.tokenize(sentence)):
            #Pokud se jedná o nový POS tag, přidá se do mapy.
            #Pokud se jedná o již existující POS tag, token se přidá do seznamu tokenů pro daný POS tag.
            if pos_tag not in pos_tag_word_map.keys():
                pos_tag_word_map[pos_tag] = set()
            pos_tag_word_map[pos_tag].add(token)
    #Převede mapu na seznam pro snadnější manipulaci.
    pos_tag_word_map_list = {}
    for pos_tag in pos_tag_word_map.keys():
        pos_tag_word_map_list[pos_tag] = list(pos_tag_word_map[pos_tag])
    return pos_tag_word_map_list

def get_augmented_dataset(augmentation_params, dataset, pos_tag_word_map_list, tokenizer=BasicTokenizer, include_idx=True):
    """Generates augmented dataset based on the given parameters."""
    #Generuje augmentovaný dataset na základě daných parametrů.
    iters = []
    #Dataset projde několikrát dle vstupního parametru.
    for _ in range(augmentation_params['n_iter']):
        data = {}
        #Pro zachování všech vstupních sloupců.
        for column in dataset.column_names:
          data[column] = []
        try:
          #Pro každý záznam v datasetu.
          for row in dataset:
            res = []
            #Provedení tokenizace vstupu a získání POS tagu.
            for word, pos_tag in nltk.pos_tag(tokenizer.tokenize(row['sentence'])):
              X = random.uniform(0,1)
              #Pokud je splněna podmínka pro maskování, dojde k zamaskování tokenu.
              if X < augmentation_params['p_mask']:
                res.append('[MASK]')
              #Pokud je splněna podmínka pro nahrazení tokenu, dojde k nahrazení tokenu jiným tokenem z daného POS tagu.
              elif X < augmentation_params['p_pos']:
                res.append(random.choice(pos_tag_word_map_list[pos_tag]))
              #Jinak dojde k zachování původního tokenu.
              else:
                res.append(word)
            #Pokud je splněna podmínka zkrácení výstupu, dopjde k jeho zkrácení.
            if random.uniform(0,1) < augmentation_params['p_ng']:
              #Minimální a maximální délka výstupu.
              n_gram_length = random.randint(20, 70)
              start = random.randrange(max(1, len(res)-n_gram_length))
              res = res[start: start+n_gram_length]
            synthetic_sample = ' '.join(res)
            #Vytvoření záznamů pro vyždanované sloupce.
            data['sentence'].append(synthetic_sample)
            data['idx'].append(row["idx"]) if include_idx else ""
            data['label'].append(row['label'])
          #Přidání výstupu celého jednoho průchodu do seznamu průchodů.
          iters.append(data)
        except Exception as e:
            print(e)
    return iters

def get_vocab(dataset):
    """Generates a vocabulary from the given dataset."""
    #Generuje slovník z daného datasetu (všechny unikátní tokeny).
    all_tokens = []
    for data in dataset:
        for token in data:
            all_tokens.append(token)
    vocab = set(all_tokens)
    return vocab

def get_embeddings_indeces(glove_file_path):  
    """Loads GloVe embeddings from the given file."""
    #Načtení GloVe embeddingů ze zadaného souboru.
    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index

def get_embedding_matrix(num_tokens, embedding_dim, word_index, embeddings_index):
    """Generates an embedding matrix from the given word index and embeddings."""
    #Vytvoření embedding matice z daného indexu slov a embeddingů. Výpočet počtu nalezených a nenalezených tokenů v embeddinzích.
    hits = 0
    misses = 0
    #Inicializace matice nulami.
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    #Pro každý token v indexu slov dojde k pokusu o nalezení embeddingu v embeddingovém indexu.
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        #Pokud je nalezen, dojde k jeho uložení do embedding matice.
        if embedding_vector is not None and len(embedding_vector) != 0:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    print(f"Converted {hits} words ({misses}) misses")
    return embedding_matrix

def padd(data, max_length):
    """Pads the given data to the specified maximum length."""
    #Padding pro daná data na maximální délku.
    padding_length = max_length - len(data)
    if padding_length > 0:
        padding = [0 for _ in range(padding_length)]
        data.extend(padding)
    return data[:max_length]

def generate_real_test_file_sst2(logits, filename):
    """Generates a test file for the SST-2 dataset to be uploaded to GLUE Benchmark."""
    #Generuje testovací soubor pro dataset SST-2, který má být nahrán na GLUE benchmark.
    labels = []
    #Definovaná struktura dle GLUE Benchmark.
    labels.append("id\tlabel\n")
    #Provedení určení třídy na základě logitů.
    for index, logit in enumerate(logits):
        labels.append(f"{index}\t{torch.topk(torch.as_tensor(logit), k=1).indices.numpy()[0]}\n")
    #Vytvoření souboru pro stažení.
    with open(filename, "w") as file:
        file.writelines(labels)
    print(f"Created output file named: {filename} upload it to GLUE benchmark to obtain results!")

class BenchMarkRunner:
    """Benchmark runner for measuring inference speed."""
    #Běh benchmarku pro měření rychlosti inference.
    def __init__(self, model, data_loader, device, num_tries):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.num_tries = num_tries

    def infer_speed_comp(self):
        for batch in self.data_loader:
            with torch.no_grad():
                _ = self.model(**batch)
            break

    def run_benchmark(self):    
        timer = benchmark.Timer(
                stmt="self.infer_speed_comp()",
                globals={"self": self},
                num_threads=torch.get_num_threads(),
            )
        
        return timer.timeit(self.num_tries)
    
def get_scores(dataset):
    """Computes F1, Accuracy, Precision and Recall scores for the given dataset."""
    #Vypočítání F1, Accuracy, Precision a Recall skóre pro daný dataset.
    preds = []
    for val in dataset:
        preds.append(torch.topk(val["logits"], k=1).indices.numpy()[0])
    
    f1 = f1_score(dataset["labels"].numpy(), preds, average="macro")
    acc = accuracy_score(dataset["labels"].numpy(), preds)
    precision = precision_score(dataset["labels"].numpy(), preds, average="macro")
    recall = recall_score(dataset["labels"].numpy(), preds, average="macro")
    
    print(f"F1 score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")