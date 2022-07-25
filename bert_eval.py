# Baseline model based on BERT (evaluation script).
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from pathlib import Path
from typing import List

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sentence_transformers import SentenceTransformer, util

from bert import eval_model, my_eval_model


if __name__ == '__main__':
    # data_path = Path('/data1/nzw/Corpus/zhlb3/')
    # model_path = Path('/data1/nzw/model/zhlb3/')
    # valid_ids: List[int] = list(range(251, 301))

    # tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
    # model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
    #     model_path)

    # model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # acc: float
    # mrr: float
    # acc, mrr = eval_model(data_path, valid_ids, tokenizer, model, return_mrr=True)
    # print(f'acc: {acc:.3f}')
    # print(f'mrr: {mrr:.3f}')

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # txts = [
    #     '原神的出现与火爆，对现今的国产游戏市场进行了一次强有力的刺激，促进了国产游戏走出畸形的商业道路。',
    #     '单拎一个节点实际上是对其他重要事情的一种忽视。您方今天其实需要论证的是2020年9月18日原神出现之前，国产游戏都是全部黑暗的，',
    #     '所以说您方把过于抬高原神的价值反而不具有普适性。',
    #     '我方既然认为您方所举的那些例子，2019年中国新增的游戏相关企业就已经超过6.6.5万家，而且早在2017年开始自结合b站等等就开始圈地跑马，所以游戏产业早就火热起来了。',
    #     '我方反而觉得原神的高投入可能会成为一种限制，比如说米哈游为了研发原神投入了1亿美金，未来每年还要投入两只3亿美金，这样的话这样的经验对于一些中小厂商来说完全没有借鉴意义，对于一些中小厂商来说，也许太无绘卷和中国式家长这种独立的游戏才是他们可以学习的范本。',
    #     '您方今天需要论证的是在当时的火热下还是出的都是烂游戏，但我方告诉你腾讯已经出了像骊山萨满或者是墨剑这样很好的游戏，所以说您方并不能证明说当时的投就完全是为了做烂游戏，而我方已经证明腾讯这样的大厂都已经在向好做高质量的游戏，谢谢。'
    # ]
    # vecs = []
    # for txt in txts:
    #     vecs.append(model.encode(txt, convert_to_tensor=True))
    # sim_score = []
    # for i in range(1,6):
    #     sim_score.append(util.pytorch_cos_sim(vecs[0], vecs[i]))
    # print('sim_score', sim_score)

    data_path = Path('/data1/nzw/Corpus/zhlb3/')
    model_path = Path('/data1/nzw/model/sn-xlm-roberta-base/')
    valid_ids: List[int] = list(range(251, 301))

    acc: float
    mrr: float
    model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    acc, mrr = my_eval_model(data_path, valid_ids, model)
    print(f'acc: {acc:.3f}')
    print(f'mrr: {mrr:.3f}')
