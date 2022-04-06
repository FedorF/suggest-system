from typing import Dict, List, Optional, Tuple

from langdetect import detect

from src.knrm import KNRM


def is_english(text: str):
    if detect(text) == 'en':
        return True
    else:
        return False


class QueryHandler:
    def __init__(self):
        self._model_is_ready = False
        self._index_size = None
        self.model = KNRM()

    @property
    def model_is_ready(self):
        return self._model_is_ready

    @property
    def index_size(self):
        return self._index_size

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
