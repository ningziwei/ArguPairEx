# Utilities for baseline models.
#
# Author: Jian Yuan
# Modified: Jingcong Liang

from pathlib import Path
from typing import Dict, List

import numpy as np


def str2dict(txt: str) -> Dict[str, str]:
    '''
    cq: 目标论点上下文
    cr: 互动论点上下文
    q: 目标论点
    r: 互动论点
    '''
    txt_list = txt.split('\n\n')[:-1]
    # print([t[:30] for t in txt_list])
    txt_list = [t.split('：\n')[1].replace('\n','') for t in txt_list]
    return dict(zip(('cq', 'cr', 'q', '1', '2', '3', '4', '5'), txt_list))


def get_labels(data_path: Path, ids: List[int]) -> np.ndarray:
    return np.array([int((data_path / f'{x}.txt').open(encoding='gb18030').read()[-1])
                     for x in ids])


def compute_acc(label: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(label == np.argmax(pred, axis=1) + 1)


def compute_mrr(label: np.ndarray, pred: np.ndarray) -> float:
    rank: np.ndarray = np.argsort(-pred, axis=1).argsort(axis=1)
    return np.mean(1 / (1 + rank[np.arange(label.shape[0]), label - 1]))
