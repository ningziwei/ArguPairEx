# Baseline model based on BERT (training script).
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from pathlib import Path
from typing import List

import torch
from transformers import set_seed
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from bert import *

if __name__ == '__main__':
    '''base model'''
    # set_seed(2022)
    # data_path = Path('/data1/nzw/Corpus/zhlb3/')
    # model_path = Path('/data1/nzw/model_saved/zhlb3/')

    # train_ids: List[int] = list(range(1, 251))
    # valid_ids: List[int] = list(range(251, 301))

    # model_card = 'bert-base-chinese'
    # tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_card)
    # dataset = Track3Dataset(data_path, train_ids, 'train', tokenizer)
    # # for _ in dataset: continue
    # # for _ in dataset: continue
    # # for _ in dataset: continue
    # model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
    #     model_card, num_labels=2)
    # model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # train(data_path, train_ids, valid_ids, tokenizer, model, model_path)

    '''try model'''
    # #Define the model. Either from scratch of by loading a pre-trained model
    # model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')

    # #Define your train examples. You need more than just two examples...
    # train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    #     InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

    # #Define your train dataset, the dataloader and the train loss
    # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
    # train_loss = losses.CosineSimilarityLoss(model)

    # #Tune the model
    # model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

    '''my model'''
    # data_path = Path('/data1/nzw/Corpus/zhlb3/')
    # train_ids: List[int] = list(range(1, 251))
    # valid_ids: List[int] = list(range(251, 301))
    # model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')
    # # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # # print(model.encode(['我们的','我们的分工']))
    # train_2(data_path, train_ids, valid_ids, model)

    from my_model import APE
    data_path = Path('/data1/nzw/Corpus/zhlb3/')
    train_ids: List[int] = list(range(1, 251))
    valid_ids: List[int] = list(range(251, 301))
    # encoder = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = APE(encoder.get_sentence_embedding_dimension())
    model.cuda()
    train_3(data_path, train_ids, valid_ids, model, encoder)


    
