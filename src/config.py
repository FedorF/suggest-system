from pathlib import Path

from pydantic import BaseSettings


class DevConfig(BaseSettings):
    EMB_PATH_KNRM: Path = Path('/binary/')
    VOCAB_PATH: Path = Path('/binary/')
    MLP_PATH: Path = Path('/binary/')
    EMB_PATH_GLOVE: Path = Path('/binary/')


class ProdConfig(BaseSettings):
    EMB_PATH_KNRM: Path = Path('/')
    VOCAB_PATH: Path = Path('/')
    MLP_PATH: Path = Path('/')
    EMB_PATH_GLOVE: Path = Path('/')
