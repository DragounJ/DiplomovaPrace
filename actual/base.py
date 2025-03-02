from transformers import Trainer, TrainingArguments, MobileNetV2Config, MobileNetV2ForImageClassification, BasicTokenizer
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import Dataset
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
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data.append(dict[b'data'])
                self.targets.extend(dict[b'fine_labels'])
                self.logits.extend(dict[b'logits'])  
                self.logits_aug.extend(dict[b'logits_aug'])   
        elif self.dataset_part == dataset_part.TEST:
            data_file = os.path.join(self.root, 'cifar-100-python', 'test')
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data.append(dict[b'data'])
                self.targets.extend(dict[b'fine_labels'])
                self.logits.extend(dict[b'logits'])  
        else:
            data_file = os.path.join(self.root, 'cifar-100-python', 'eval')
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data.append(dict[b'data'])
                self.targets.extend(dict[b'fine_labels'])
                self.logits.extend(dict[b'logits'])  
            
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

        #self.labels = nn.functional.one_hot(torch.as_tensor(self.labels),10)

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


def generate_logits(dataloader, model, images=True):
    """Generates logits for given input."""
    logits_arr = []
    for batch in tqdm(dataloader, desc="Generating logits for given dataset: "):
        
        with torch.no_grad():
            if images:
                pixel_values, labels = batch
                outputs = model(pixel_values)
            else:
                outputs = model(**batch)

            logits = outputs["logits"]
        logits_arr.append(logits.cpu().numpy())

    logits_arr_flat = []
    for tensor in logits_arr:
        logits_arr_flat.extend(tensor)
    return logits_arr_flat



def check_acc(dataset, desc="Accuracy for given set is: "):
    """Swift Acc checker for given dataset and its appended logits."""
    corr = []
    for val in tqdm(dataset, desc="Calculating accuracy based on the saved logits: "): 
        if torch.topk(val["logits"], k=1).indices.numpy()[0] == val["labels"]:  corr.append(True)
    
    return(f"{desc} {len(corr)/len(dataset)}")  


def remove_diff_pred_class(normal, aug, pytorch_dataset=True):
    """Removes those entries from aug, that do not have the same biggest logit as normal."""
    rem_ls = []
    for index, val in tqdm(enumerate(aug), total=len(aug), desc="Removing entries from augmented dataset that are different from the base one - based on saved logits: "):
        target_alt = torch.topk(val["logits"], k=1).indices.numpy()[0]
        target_act = torch.topk(normal[index]["logits"], k=1).indices.numpy()[0]
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
    """Custom wrapper of training args for distillation."""
    def __init__(self, lambda_param, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.lambda_param = lambda_param
        self.temperature = temperature


def get_training_args(output_dir, logging_dir, remove_unused_columns=True, lr=5e-5, epochs=5, weight_decay=0, lambda_param=.5, temp=5, batch_size=128, num_workers=4):
    """Returns training args that can be adjusted."""
    return (
        Custom_training_args(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr, #Defaultní hodnota 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        seed = 42,  #Defaultní hodnota 
        metric_for_best_model="f1",
        load_best_model_at_end = True,
        fp16=True, 
        logging_dir=logging_dir,
        remove_unused_columns=remove_unused_columns,
        lambda_param = lambda_param, 
        temperature = temp,
        dataloader_num_workers=num_workers
    ))


def get_random_init_mobilenet(num_labels):
    """Returns randomly initialized MobileNetV2."""
    reset_seed(42)
    student_config = MobileNetV2Config()
    student_config.num_labels = num_labels
    return MobileNetV2ForImageClassification(student_config)


def get_mobilenet(num_labels):
    """Returns initialized MobileNetV2."""
    model_pretrained = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    in_features = model_pretrained.classifier.in_features

    model_pretrained.classifier = nn.Linear(in_features,num_labels) #Úprava klasifikační hlavy
    model_pretrained.num_labels = num_labels
    model_pretrained.config.num_labels = num_labels

    return model_pretrained


def freeze_model(model):
    """Freezes all params apart from classification head."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

class DistilTrainer(Trainer):
    """Distilation trainer, computes loss with logits from teacher in mind. Logits are precomputed."""
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
        soft_student = F.log_softmax(student_output['logits'] / self.temperature, dim=-1)

        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = student_output["loss"]

        loss = ((1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss)
        return (loss, student_output) if return_outputs else loss
    
def count_parameters(model):
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
    dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', return_tensors="pt", max_length=60), batched=True, desc="Tokenizing the provided dataset")
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', "attention_mask"], device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return dataset


class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, fc_dim, output_dim, embedding_matrix, freeze_embed = True):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embed)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)  
        self.dropout = nn.Dropout(.2)
        self.fc2 = nn.Linear(fc_dim, output_dim)

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)  
        _, (h_n, _) = self.lstm(embedded)
        h_forward = h_n[-2, :, :]  # Last forward hidden state
        h_backward = h_n[-1, :, :]  # Last backward hidden state
        out_cat = torch.cat((h_forward, h_backward), dim=1)
        fc1_out = F.relu(self.fc1(out_cat))
        dropped = self.dropout(fc1_out)
        logits = self.fc2(dropped)
        
        if labels is not None:
            labels = nn.functional.one_hot(labels, num_classes=self.fc2.out_features) 
            loss_fn = nn.CrossEntropyLoss() 
            loss = loss_fn(logits, labels.float())
            return {"loss" : loss, "logits" : logits}
        return {"loss" : None, "logits": logits}
    

def get_pos_tag_word_map(sentences, tokenizer=BasicTokenizer(do_lower_case=True)):
    pos_tag_word_map = {}
    for sentence in sentences:
        for token, pos_tag in nltk.pos_tag(tokenizer.tokenize(sentence)):
            if pos_tag not in pos_tag_word_map.keys():
                pos_tag_word_map[pos_tag] = set()
                pos_tag_word_map[pos_tag].add(token)

    pos_tag_word_map_list = {}
    for pos_tag in pos_tag_word_map.keys():
        pos_tag_word_map_list[pos_tag] = list(pos_tag_word_map[pos_tag])
    return pos_tag_word_map_list

def get_augmented_dataset(augmentation_params, dataset, pos_tag_word_map_list, tokenizer=BasicTokenizer, include_idx=True):
    iters = []
    for _ in range(augmentation_params['n_iter']):
        data = {}
        for column in dataset.column_names:
          data[column] = []
        try:
          for row in dataset:
            res = []
            for word, pos_tag in nltk.pos_tag(tokenizer.tokenize(row['sentence'])):
              X = random.uniform(0,1)
              if X < augmentation_params['p_mask']:
                res.append('[MASK]')
              elif X < augmentation_params['p_pos']:
                res.append(random.choice(pos_tag_word_map_list[pos_tag]))
              else:
                res.append(word)
            if random.uniform(0,1) < augmentation_params['p_ng']:
              n_gram_length = random.randint(20, 70)
              start = random.randrange(max(1, len(res)-n_gram_length))
              res = res[start: start+n_gram_length]
            synthetic_sample = ' '.join(res)
            data['sentence'].append(synthetic_sample)
            data['idx'].append(row["idx"]) if include_idx else ""
            data['label'].append(row['label'])
          iters.append(data)
        except Exception as e:
            print(e)
    return iters

def get_vocab(dataset):
    all_tokens = []
    for data in dataset:
        for token in data:
            all_tokens.append(token)
    vocab = set(all_tokens)
    return vocab

def get_embeddings_indeces(glove_file_path):  
    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index

def get_embedding_matrix(num_tokens, embedding_dim, word_index, embeddings_index):
    hits = 0
    misses = 0
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and len(embedding_vector) != 0:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    print(f"Converted {hits} words ({misses}) misses")
    return embedding_matrix

def padd(data, max_length):
    padding_length = max_length - len(data)
    if padding_length > 0:
        padding = [0 for _ in range(padding_length)]
        data.extend(padding)
    return data[:max_length]

def generate_real_test_file_sst2(logits, filename):
    labels = []
    labels.append("id\tlabel\n")
    for index, logit in enumerate(logits):
        labels.append(f"{index}\t{torch.topk(torch.as_tensor(logit), k=1).indices.numpy()[0]}\n")

    with open(filename, "w") as file:
        file.writelines(labels)
    print(f"Created output file named: {filename} upload it to GLUE benchmark to obtain results!")