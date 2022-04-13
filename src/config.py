from pathlib import Path
import os

from pydantic import BaseSettings


class DevConfig(BaseSettings):
    EMB_PATH_KNRM: Path = Path('/binary/embed.binary')
    VOCAB_PATH: Path = Path('/binary/vocab.json')
    MLP_PATH: Path = Path('/binary/mlp.binary')
    EMB_PATH_GLOVE: Path = Path('/binary/glove.6B.50d.txt')


class ProdConfig(BaseSettings):
    EMB_PATH_KNRM: Path = os.environ['EMB_PATH_KNRM']
    VOCAB_PATH: Path = os.environ['VOCAB_PATH']
    MLP_PATH: Path = os.environ['MLP_PATH']
    EMB_PATH_GLOVE: Path = os.environ['EMB_PATH_GLOVE']
