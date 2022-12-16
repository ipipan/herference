from pathlib import Path
from spacy.util import load_model_from_init_py

from herference.manager import Herference
from herference.coref_sp import make_herference


def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)
