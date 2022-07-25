# Baseline model based on BERT (core module).
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import choice
from typing import ClassVar, DefaultDict, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torch.optim as optim
from tqdm.auto import tqdm
from torch.optim import Adam, AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, BertForSequenceClassification, BertTokenizer

from util import compute_acc, compute_mrr, get_labels, str2dict


@dataclass
class Track3Dataset(Dataset):
    data_path: Path
    ids: List[int]
    mode: str
    tokenizer: BertTokenizer

    _TAGS: ClassVar = '12345'

    def __post_init__(self) -> None:
        assert self.mode in ('train', 'test')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[BatchEncoding, Optional[List[int]]]:
        '''
        目标论点分别跟互动论点和一个无关论点拼成句子
        tokenizer(['阿斯顿','阿斯顿'],['他们说要','戒掉你的狂'],truncation=True)
        {
            'input_ids': [
                [101, 7350, 3172, 7561, 102, 800, 812, 6432, 6206, 102], 
                [101, 7350, 3172, 7561, 102, 2770, 2957, 872, 4638, 4312, 102]], 
            'token_type_ids': [
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], 
            'attention_mask': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        }
        '''
        s: str = (self.data_path / f'{self.ids[idx]}.txt').open(encoding='gb18030').read()
        record: Dict[str, str] = str2dict(s)
        answer: Optional[str] = s[-1] if '互动论点：' in s else None

        first_text: List[str]
        second_text: List[str]
        label: Optional[List[int]]

        if self.mode == 'train':
            first_text = [record['q'], record['q']]
            second_text = [record[answer], record[choice(self._TAGS.replace(answer, ''))]]
            label = [1, 0]
        else:
            first_text = [record['q'] for _ in self._TAGS]
            second_text = [record[tag] for tag in self._TAGS]
            label = [int(tag == answer) for tag in self._TAGS] if answer is not None else None

        return self.tokenizer(first_text, second_text, truncation=True), label

def create_mini_batch(
    samples: List[Tuple[BatchEncoding, Optional[List[int]]]]
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    all_tokens: DefaultDict[str, List[List[int]]] = defaultdict(list)
    all_labels: List[int] = []

    for tokens, labels in samples:
        for k, v in tokens.items():
            all_tokens[k].extend(v)
        if labels is not None:
            all_labels.extend(labels)

    return {k: pad_sequence([torch.tensor(x) for x in v], batch_first=True) for k, v in
            all_tokens.items()}, torch.tensor(all_labels) if len(all_labels) > 0 else None

def inputs_to_device(inputs: Dict[str, torch.Tensor],
                     device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in inputs.items()}

def get_predictions(data_path: Path, ids: List[int], tokenizer: BertTokenizer,
                    model: BertForSequenceClassification) -> np.ndarray:
    dataset = Track3Dataset(data_path, ids, 'test', tokenizer)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=create_mini_batch)
    pred: List[np.ndarray] = []

    with torch.no_grad():
        model.eval()
        for inputs, _ in loader:
            inputs: Dict[str, torch.Tensor] = inputs_to_device(inputs, model.device)
            outputs: torch.Tensor = model(**inputs)
            # print('104', outputs)
            # print('105', outputs[0])
            # print('106', outputs[0][:, 1])
            pred.append(outputs[0][:, 1].cpu().numpy())

    return np.stack(pred)

def eval_model(data_path: Path, ids: List[int], tokenizer: BertTokenizer,
               model: BertForSequenceClassification,
               return_mrr: bool = False) -> Union[float, Tuple[float, float]]:
    label: np.ndarray = get_labels(data_path, ids)
    pred: np.ndarray = get_predictions(data_path, ids, tokenizer, model)
    acc: float = compute_acc(label, pred)
    # print('label', label)
    # print('pred', pred)

    if return_mrr:
        mrr: float = compute_mrr(label, pred)
        return acc, mrr
    else:
        return acc

def train(data_path: Path, train_ids: List[int], valid_ids: List[int], tokenizer: BertTokenizer,
          model: BertForSequenceClassification, model_path: Path) -> None:
    # dataset = Track3Dataset(data_path, train_ids, 'train', tokenizer)
    # loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=create_mini_batch)
    optimizer = AdamW(model.parameters(), lr=2e-6, weight_decay=0.1)
    # optimizer = optim.SGD(model.parameters(),lr=2e-6,momentum=0.9,weight_decay=0)

    best_acc: float = eval_model(data_path, valid_ids, tokenizer, model)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    for epoch in range(50):
        model.train()
        running_loss: float = 0.0
        dataset = Track3Dataset(data_path, train_ids, 'train', tokenizer)
        loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=create_mini_batch)

        for inputs, labels in tqdm(loader):
            # print('133 inputs', inputs)
            # print('inputs', len(inputs['input_ids']))
            # print('inputs', inputs['input_ids'].shape)
            # print('labels', labels)
            inputs: Dict[str, torch.Tensor] = inputs_to_device(inputs, model.device)
            labels: torch.Tensor = labels.to(model.device)

            optimizer.zero_grad()
            print('labels', labels)
            loss: torch.Tensor = model(**inputs, labels=labels)[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc: float = eval_model(data_path, valid_ids, tokenizer, model)
        print(f'[epoch {epoch + 1}] loss: {running_loss:.3f}, acc: {acc:.3f} best: {best_acc:.3f}')

        if acc >= best_acc:
            best_acc = acc
            model.save_pretrained(model_path)
            print('current best model saved')

    print(f'best acc: {best_acc:.3f}')

from time import time
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import InputExample, losses, evaluation

@dataclass
class Track3Dataset_2(Dataset):
    data_path: Path
    ids: List[int]
    mode: str

    def __post_init__(self) -> None:
        assert self.mode in ('train', 'test')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        '''
        需要进一步修改确保数据解析不出错
        record: {'cq':'', 'cr':'', 'q':'', '1':'', ..., '5':''}
        answer: '2'
        '''
        # print(idx)
        s: str = (self.data_path / f'{self.ids[idx]}.txt').open(encoding='gb18030').read()
        record: Dict[str, str] = str2dict(s)
        answer: Optional[str] = s[-1] if '互动论点：' in s else None
        return record, answer

def get_sim_score(model, txts):
    vecs = []
    for txt in txts:
        vecs.append(model.encode(txt))
    sim_score = []
    for i in range(1,6):
        sim_score.append(util.pytorch_cos_sim(vecs[0], vecs[i]))
    return sim_score

def get_sim_score_(model, txts):
    vec = model.encode(txts)
    norm = np.linalg.norm(vec,axis=-1)
    vec = vec/np.expand_dims(norm,-1)
    src = vec[:1,:]
    cand = vec[1:,:]
    score = np.dot(src, cand.T)
    score = score.squeeze()
    return score

def eval_model_2(data_path, ids, model):
    label: list = []
    pred: list = []
    scores: list = []
    dataset = Track3Dataset_2(data_path, ids, 'test')
    # tic = time()
    for data, answer in dataset:
        # print([data[n] for n in 'q12345'])
        txts = [data[n] for n in 'q12345']
        # score = get_sim_score(model, txts)
        score = get_sim_score_(model, txts)
        # print(np.array(score), np.array(score_))
        label.append(int(answer))
        pred.append(np.argmax(score)+1)
        scores.append(score)
    # toc = time()
    # print('用时', toc-tic)
    # print('label', label)
    # print('pred', pred)
    label = np.array(label)
    pred = np.array(pred)
    scores = np.array(scores)
    acc: float = compute_acc(label, scores)
    mrr: float = compute_mrr(label, scores)
    return acc, mrr

_TAGS = '12345'
def gen_dataset_2(model, dataset, soft_sim=False, label_type=float):
    '''
    生成训练数据
    给句子对添加相似度标签
    '''
    pairs = []
    if soft_sim:
        for data, answer in dataset:
            txts = [data[n] for n in 'q12345']
            # score = get_sim_score(model, txts)
            score = get_sim_score_(model, txts) - 0.2
            score[int(answer)-1] = np.max(score) + 0.4
            score = np.minimum(1.0, score)
            score = np.maximum(0.0, score)
            pairs_ = [[[txts[0],txts[i]],score[i-1]] for i in range(1,6)]
            pairs_ = [
                pairs_[int(answer)-1], 
                pairs_[int(choice(_TAGS.replace(answer, '')))-1]
            ]
            pairs += pairs_
            # print(score, pairs[-2:])
    else:
        for data, answer in dataset:
            txts = [data[n] for n in 'q12345']
            pairs_ = [[[txts[0],txts[i]],0.0] for i in range(1,6)]
            pairs_[int(answer)-1][-1] = 1.0
            pairs_ = [
                pairs_[int(answer)-1], 
                pairs_[int(choice(_TAGS.replace(answer, '')))-1]
            ]
            pairs += pairs_
    pairs = [InputExample(texts=t[0], label=label_type(t[1])) for t in pairs]
    return pairs

def train_2(data_path, train_ids, valid_ids, model):
    dataset = Track3Dataset_2(data_path, train_ids, 'train')
    # train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    #     InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
    # train_examples = gen_dataset(model, dataset, soft_sim=True)
    # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
    loss_type = 'cos'
    if loss_type == 'cos':
        train_loss = losses.CosineSimilarityLoss(model)
        label_type = float
    else:
        train_loss = losses.SoftmaxLoss(model,
            sentence_embedding_dimension=6768, num_labels=2)
        label_type = int
    
    acc: float = eval_model_2(data_path, valid_ids, model)
    print(acc)
    # evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    # Tune the model
    lr = 2e-5
    for i in range(100):
        train_examples = gen_dataset_2(model, dataset, soft_sim=False, label_type=label_type)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=5)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)], 
            epochs=1, warmup_steps=10, optimizer_params={'lr':lr}
        )
        if (i+1)%50==0: lr *= 0.1
        acc: float = eval_model_2(data_path, valid_ids, model)
        print(i, acc)

'''以分类的方式使用SBERT'''
@dataclass
class Track3Dataset_3(Dataset):
    data_path: Path
    ids: List[int]
    mode: str

    _TAGS: ClassVar = '12345'

    def __post_init__(self) -> None:
        assert self.mode in ('train', 'test')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        '''
        '''
        s: str = (self.data_path / f'{self.ids[idx]}.txt').open(encoding='gb18030').read()
        record: Dict[str, str] = str2dict(s)
        answer: Optional[str] = s[-1] if '互动论点：' in s else None

        first_text: List[str]
        second_text: List[str]
        label: Optional[List[int]]

        if self.mode == 'train':
            first_text = [record['q'], record['q']]
            second_text = [record[answer], record[choice(self._TAGS.replace(answer, ''))]]
            label = [1, 0]
        else:
            first_text = [record['q'] for _ in self._TAGS]
            second_text = [record[tag] for tag in self._TAGS]
            label = [int(tag == answer) for tag in self._TAGS] if answer is not None else None

        return first_text, second_text, label

def gen_dataset_3(model, dataset):
    '''
    生成训练数据
    给句子对添加相似度标签
    '''
    pairs = []
    for data, answer in dataset:
        txts = [data[n] for n in 'q12345']
        pairs_ = [[txts[0],txts[i]] for i in range(1,6)]
        if answer is not None:
            pairs_ = [
                [model.encode(pairs_[int(answer)-1]), 1],
                [model.encode(pairs_[int(choice(_TAGS.replace(answer, '')))-1]), 0]
            ]
        else:
            pairs_ = [
                [model.encode(pairs_[int(answer)-1]), None],
                [model.encode(pairs_[int(choice(_TAGS.replace(answer, '')))-1]), None]
            ]
        pairs += pairs_
    return pairs

def create_mini_batch_3_(samples):
    all_emb1 = []
    all_emb2 = []
    all_labels = []
    # print('samples', samples)
    for pair in samples:
        all_emb1.append(pair[0][0])
        all_emb2.append(pair[0][1])
        if pair[1] is not None:
            all_labels.append(pair[1])
    return torch.Tensor(all_emb1), torch.Tensor(all_emb2),\
         torch.tensor(all_labels) if len(all_labels) > 0 else None

def create_mini_batch_3(samples):
    all_txt1 = []
    all_txt2 = []
    all_labels = []
    # print('samples', samples)
    for txt1, txt2, labels in samples:
        all_txt1 += txt1
        all_txt2 += txt2
        all_labels += labels
    # print('381', len(all_txt1), len(all_txt2), len(all_labels))
    all_labels = torch.tensor(all_labels) if len(all_labels) > 0 else None
    return all_txt1, all_txt2, all_labels

def eval_model_3(data_path, ids, model, encoder):
    label: np.ndarray = get_labels(data_path, ids)
    pred: list = []
    dataset = Track3Dataset_3(data_path, ids, 'test')
    loader = DataLoader(dataset=dataset, batch_size=1, 
        shuffle=False, collate_fn=create_mini_batch_3)
    pred: List[np.ndarray] = []

    with torch.no_grad():
        model.eval()
        for txt1, txt2, _ in loader:
            # print('400', len(txt1), len(txt2))
            emb1 = torch.Tensor(encoder.encode(txt1))
            emb2 = torch.Tensor(encoder.encode(txt2))
            emb1 = emb1.to(encoder.device)
            emb2 = emb2.to(encoder.device)
            outputs = model(emb1, emb2).squeeze()
            # print('400', outputs.shape)
            # 预测时只比较候选论点在类别1上的相对大小
            pred.append(outputs[:, 1].cpu().numpy())
    pred = np.array(pred)
    # print(pred.shape)
    pred = pred.reshape((-1,5))
    # print(label, pred)
    acc: float = compute_acc(label, pred)
    mrr: float = compute_mrr(label, pred)
    return acc, mrr

def train_3(data_path, train_ids, valid_ids, model, encoder) -> None:
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.1)
    best_acc, _ = eval_model_3(data_path, valid_ids, model, encoder)

    for epoch in range(200):
        model.train()
        running_loss: float = 0.0
        train_examples = Track3Dataset_3(data_path, train_ids, 'train')
        train_dataloader = DataLoader(
            train_examples, shuffle=True, 
            batch_size=5, collate_fn=create_mini_batch_3)

        for txt1, txt2, labels in tqdm(train_dataloader):
            emb1 = torch.Tensor(encoder.encode(txt1))
            emb2 = torch.Tensor(encoder.encode(txt2))
            emb1 = emb1.to(encoder.device)
            emb2 = emb2.to(encoder.device)
            labels = labels.to(encoder.device)

            optimizer.zero_grad()
            loss: torch.Tensor = model(emb1, emb2, labels=labels)[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc, mrr = eval_model_3(data_path, valid_ids, model, encoder)
        print(f'[epoch {epoch + 1}] loss: {running_loss:.3f}, acc: {acc:.3f} mrr: {mrr:.3f} best: {best_acc:.3f}')

        if acc>best_acc:
            best_acc = acc


    # print(f'best acc: {best_acc:.3f}')

