import json
from typing import Dict, List, Optional, Tuple

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
    def __init__(self, path_embed, path_mlp, path_vocab):
        self._model_is_ready = False
        self._index_size = None

        self.path_embed = path_embed
        self.path_mlp = path_mlp
        self.path_vocab = path_vocab

        self.model, self.vocab, self.unk_words = self.build_knrm_model()

    @property
    def model_is_ready(self):
        return self._model_is_ready

    @property
    def index_size(self):
        return self._index_size

    def build_knrm_model(self):
        embed = torch.load(self.path_embed)['weight']
        mlp = torch.load(self.path_mlp)
        model = KNRM(embed, mlp)

        vocab = read_json(self.path_vocab)

        return model, vocab, unk_words

    def get_suggestion(self, queries: List[str]) -> Tuple[List[bool], List[Optional[List[Tuple[str, str]]]]]:
        lang_check, suggestions = [], []
        for query in queries:
            check = is_english(query)
            lang_check.append(check)
            if not check:
                suggestions.append(None)
            else:
                suggestions = self.model.predict(query)

        return lang_check, suggestions

    def update_index(self, documents: Dict[str, str]):
        pass
