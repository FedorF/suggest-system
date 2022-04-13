from collections import Counter
import json
import string
from typing import Dict, List, Optional, Tuple

import faiss
import nltk
import numpy as np
import pandas as pd
from langdetect import detect
import torch

from src.knrm import KNRM


def is_english(text: str):
    if detect(text) == 'en':
        return True
    else:
        return False


def read_json(path: str):
    with open(path, 'r') as f:
        output = json.load(f)

    return output


class QueryHandler:
    def __init__(self, path_embed, path_mlp, path_vocab, num_suggest=10, num_candidates=100):
        self.num_suggest = num_suggest
        self.num_candidates = num_candidates
        self._model_is_ready = False
        self.documents = None
        self.index = None
        self._index_size = None

        self.path_embed = path_embed
        self.path_mlp = path_mlp
        self.path_vocab = path_vocab

        self.model, self.vocab, self.unk_words = self.build_knrm_model()

    @property
    def model_is_ready(self):
        return self._model_is_ready and self.index_size > 0

    @property
    def index_size(self):
        return self._index_size

    def build_knrm_model(self):
        embed = torch.load(self.path_embed)['weight']
        mlp = torch.load(self.path_mlp)
        model = KNRM(embed, mlp)

        vocab = read_json(self.path_vocab)
        unk_words = None  # todo
        self._model_is_ready = True
        return model, vocab, unk_words

    def _vectorize(self, query: str) -> np.array:
        query = self.simple_preproc(query)
        mask = [self.vocab.get(token, 'OOV') for token in query]
        query = np.mean(self.model.embed_weights[mask], axis=0)

        return np.array(query)

    def _find_knn(self, query: np.array, k: int) -> List[Tuple[str, str]]:
        _, ind = self.index.search(query, k=k)
        candidates = []
        for i in ind[0]:
            if i >= 0:
                candidates.append((str(i), self.documents[str(i)]))

        return candidates

    def _rank(self, candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # todo
        return candidates

    # def _rank(self, candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    #     inputs = dict()
    #     inputs['query'] = self._text_to_token_ids([query] * len(cands))
    #     inputs['document'] = self._text_to_token_ids([cand[1] for cand in candidates])
    #     scores = self.model(inputs)
    #     res_ids = scores.reshape(-1).argsort(descending=True)
    #     res_ids = res_ids[:ret_k]
    #     res = [cands[i] for i in res_ids.tolist()]
    #
    #     return res

    def suggest_candidates(self, queries: List[str]) -> Tuple[List[bool], List[Optional[List[Tuple[str, str]]]]]:
        lang_check, suggestions = [], []
        for query in queries:
            check = is_english(query)
            lang_check.append(check)
            if not check:
                suggestions.append(None)
            else:
                query = self._vectorize(query)
                candidates = self._find_knn(query, self.num_candidates)
                ranked = self._rank(candidates)
                suggestions.append(ranked)

        return lang_check, suggestions

    def hadle_punctuation(self, inp_str: str) -> str:
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')

        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.hadle_punctuation(inp_str.lower())
        inp_str = inp_str.strip()

        return nltk.word_tokenize(inp_str)

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        return {k: c for k, c in vocab.items() if c >= min_occurancies}

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        corpus = []
        for df in list_of_df:
            for col in ['text_left', 'text_right']:
                for doc in df[col].values:
                    corpus.append(doc)
        corpus = ' '.join(set(corpus))
        corpus = self.simple_preproc(corpus)
        corpus = Counter(corpus)
        corpus = self._filter_rare_words(corpus, min_occurancies)

        return list(corpus)

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        embed = {}
        with open(file_path) as f:
            for line in f.readlines():
                line = line.split(' ')
                embed[line[0]] = line[1:]

        return embed

    def update_index(self, documents: Dict[str, str]) -> None:
        self.documents = documents
        vectors = []
        for doc in self.documents.values():
            doc = self._vectorize(doc)
            vectors.append(doc)
        vectors = np.array(vectors)

        quantizer = faiss.IndexFlatL2(vectors.shape[1])
        self.index = faiss.IndexIDMap(quantizer)
        self.index.add_with_ids(vectors, np.array(list(map(int, self.documents))))
        self._index_size = self.index.ntotal
