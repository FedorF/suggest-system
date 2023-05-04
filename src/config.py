import os

from pydantic import BaseSettings


class Config(BaseSettings):
    EMB_PATH_KNRM: str = os.environ['EMB_PATH_KNRM']
    VOCAB_PATH: str = os.environ['VOCAB_PATH']
    MLP_PATH: str = os.environ['MLP_PATH']
    EMB_PATH_GLOVE: str = os.environ['EMB_PATH_GLOVE']
    SERVICE_URL: str = os.environ['SERVICE_URL']
