from pathlib import Path

from pydantic import BaseSettings


class DevConfig(BaseSettings):
    EMB_PATH_KNRM: Path = Path('/binary/embed.binary')
    VOCAB_PATH: Path = Path('/binary/vocab.json')
    MLP_PATH: Path = Path('/binary/mlp.binary')
    EMB_PATH_GLOVE: Path = Path('/binary/glove.6B.50d.txt')


class ProdConfig(BaseSettings):
    EMB_PATH_KNRM: Path = Path('/')
    VOCAB_PATH: Path = Path('/')
    MLP_PATH: Path = Path('/')
    EMB_PATH_GLOVE: Path = Path('/')
