import os
import logging

if os.getenv('HERFERENCE_LAZY_IMPORT') != '1':
    from herference.manager import Herference
    from herference.coref_sp import make_herference
    logging.warning('HERFERENCE_LAZY_IMPORT')


from pathlib import Path
from spacy.util import load_model_from_init_py


def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)
