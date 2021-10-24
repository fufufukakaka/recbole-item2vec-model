# -*- coding: utf-8 -*-
# @Time   : 2021/10/23
# @Author : Yusuke Fukasawa
# @Email  : demonstrundum@gmail.com

r"""
ItemKNN
################################################
Reference:
    Item 2 Vec-based Approach to a Recommender System
    https://www.semanticscholar.org/paper/Item-2-Vec-based-Approach-to-a-Recommender-System-Kuzmin/94aba3f1219819d60bba21ddad1af11361d790e0
"""

import numpy as np
import torch
from gensim.models import word2vec
from recbole.data.dataset import Dataset
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class Item2Vec(GeneralRecommender):
    r"""Item2Vec です"""
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset: Dataset):
        super(Item2Vec, self).__init__(config, dataset)

        assert dataset.history_user_matrix is not None
        self.history_matrix, self.history_item_value, _ = dataset.history_item_matrix()
        self.user_item_history = {}
        for idx in range(len(self.history_matrix)):
            _history = [
                str(v[0])
                for v in self.history_matrix[idx][
                    self.history_matrix[idx].nonzero()
                ].numpy()
                if v[0] != 0
            ]
            if len(_history) > 0:
                self.user_item_history[idx] = _history

        # load parameters info(TODO)
        self.item2vec = word2vec.Word2Vec(
            self.user_item_history.values(),
            vector_size=300,
            window=10000,
            alpha=0.025,
            sample=0.00001,
            epochs=100,
            ns_exponent=0.75,
            min_alpha=0.0001,
            hs=0,
            negative=5,
            sg=1,
            seed=42,
            min_count=0,
            workers=4,
        )
        # collect missing words
        all_words = set([v for v in range(self.n_items)])
        vocabs = set([int(v) for v in list(self.item2vec.wv.key_to_index.keys())])
        self.missing_words = all_words - vocabs
        self.vocab_size = len(vocabs)

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        pass  # Not Implemented(TODO)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy().astype(int)
        _result = []

        for index in range(len(user)):
            uid = user[index]
            score = [0] * self.n_items
            _score = self.item2vec.wv.most_similar(
                self.user_item_history[uid], topn=None
            )
            for idx, _s in enumerate(_score):
                score[int(self.item2vec.wv.index_to_key[idx])] = _s
            # add missing words score
            for m in self.missing_words:
                score[int(m)] = -10000
            _result.append(score)
        result = torch.from_numpy(np.array(_result)).to(self.device)
        return result
