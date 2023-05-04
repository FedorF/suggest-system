from collections import Counter
import json
import string
from typing import Dict, List, Optional, Tuple

import faiss
from langdetect import detect
import nltk
import numpy as np
import pandas as pd
import torch

from src.knrm import KNRM


def is_english_lang(text: str) -> bool:
    if detect(text) == 'en':
        return True
    else:
        return False


def read_json(path: str) -> dict:
    with open(path, 'r') as f:
        output = json.load(f)

    return output


class QueryHandler:
    def __init__(
            self,
            path_embed: str,
            path_mlp: str,
            path_vocab: str,
            n_suggest: int = 10,
            n_candidates: int = 100,
    ):
        self.n_suggest = n_suggest
        self.n_candidates = n_candidates
        self.model = None
        self.vocab = None
        self._model_is_ready = False
        self.documents = None
        self.index = None
        self._index_size = -1
        self.path_embed = path_embed
        self.path_mlp = path_mlp
        self.path_vocab = path_vocab

    @property
    def model_is_ready(self) -> bool:
        return self._model_is_ready

    @property
    def index_is_ready(self) -> bool:
        return self.index_size > 0

    @property
    def index_size(self) -> int:
        return self._index_size

    def build_knrm_model(self):
        embed = torch.load(self.path_embed)['weight']
        mlp = torch.load(self.path_mlp)
        self.model = KNRM(embed, mlp)
        self.vocab = read_json(self.path_vocab)
        self._model_is_ready = True

    def _vectorize(self, query: str) -> np.array:
        query = self._simple_preproc(query)
        mask = [self.vocab.get(token, self.vocab['OOV']) for token in query]
        query = np.array(self.model.embed_weights[mask])
        query = np.mean(query, axis=0)

        return query

    def _find_candidates(self, query: str, k: int) -> List[Tuple[str, str]]:
        query = self._vectorize(query)
        query = query[np.newaxis, ...]
        _, ind = self.index.search(query, k=k)
        candidates = []
        for i in ind[0]:
            if i >= 0:
                candidates.append((str(i), self.documents[str(i)]))

        return candidates

    def _text_to_token_ids(self, text_list: List[str]):
        tokenized = []
        for text in text_list:
            tokenized_text = self._simple_preproc(text)
            token_ind = [self.vocab.get(i, self.vocab["OOV"]) for i in tokenized_text]
            tokenized.append(token_ind)
        max_len = max(len(elem) for elem in tokenized)
        tokenized = [elem + [0] * (max_len - len(elem)) for elem in tokenized]
        tokenized = torch.LongTensor(tokenized)
        return tokenized

    def _rank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        inputs = dict()
        inputs['query'] = self._text_to_token_ids([query] * len(candidates))
        inputs['document'] = self._text_to_token_ids([cand[1] for cand in candidates])
        scores = self.model(inputs)
        res_ids = scores.reshape(-1).argsort(descending=True)
        res_ids = res_ids[:self.n_suggest]
        res = [candidates[i] for i in res_ids.tolist()]

        return res

    def _hadle_punctuation(self, inp_str: str) -> str:
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')

        return inp_str

    def _simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self._hadle_punctuation(inp_str.lower())
        inp_str = inp_str.strip()

        return nltk.word_tokenize(inp_str)

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        return {k: c for k, c in vocab.items() if c >= min_occurancies}

    def _get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        corpus = []
        for df in list_of_df:
            for col in ['text_left', 'text_right']:
                for doc in df[col].values:
                    corpus.append(doc)
        corpus = ' '.join(set(corpus))
        corpus = self._simple_preproc(corpus)
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

    def update_index(self, documents: Dict[str, str]):
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

    def suggest_candidates(
            self,
            queries: List[str]
    ) -> Tuple[List[bool], List[Optional[List[Tuple[str, str]]]]]:

        lang_check, suggestions = [], []
        for query in queries:
            eng_lang = is_english_lang(query)
            lang_check.append(eng_lang)
            if eng_lang:
                candidates = self._find_candidates(query, self.n_candidates)
                ranked = self._rank(query, candidates)
                suggestions.append(ranked)
            else:
                suggestions.append(None)

        return lang_check, suggestions
