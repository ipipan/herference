import logging
import re
from typing import Union

import yaml
import random

import numpy as np
import torch
import spacy
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from herference import api
from herference.align import align
from herference.batch import CorefLoader
from herference.config import get_config
from herference.dataset import CorefDataset, Text, Corpus
from herference.evaluator import Evaluator
from herference.model import S2E
from herference.heads import add_heads

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Herference:
    def __init__(self, _cfg=None, device=None, model=None):
        self.cfg = _cfg if _cfg is not None else get_config()
        self.cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device

        self.transformers_logger = logging.getLogger("transformers")
        self.transformers_logger.setLevel(self.cfg.LOG_LEVEL)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=self.cfg.LOG_LEVEL)

        set_seed(self.cfg.model_config.seed)

        transformers_cfg = AutoConfig.from_pretrained(
            self.cfg.MODEL_NAME
        ) if not model else None

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.MODEL_NAME
        )

        S2E.config_class = self.cfg.MODEL_CONFIG_CLASS
        S2E.base_model_prefix = self.cfg.model_config.base_model_prefix
        model = S2E.from_pretrained(self.cfg.MODEL_NAME if not model else model,
                                    config=transformers_cfg,
                                    params=self.cfg.model_config
                                    )  # without passing args

        model.to(self.cfg.device)

        self.tokenizer = tokenizer
        self.model = model
        try:
            self.nlp = spacy.load(self.cfg.SPACY_MODEL_NAME)
        except IOError as e:
            logger.info(f'Could not load Spacy model {self.cfg.SPACY_MODEL_NAME}')
            spacy.cli.download(self.cfg.SPACY_MODEL_NAME)
            self.nlp = spacy.load(self.cfg.SPACY_MODEL_NAME)

    def predict(self, data_point: Union[str, list], mention_heads: bool = True):
        if isinstance(data_point, str):
            data_point = re.sub('\n', ' ', data_point)
            data_point = data_point.split(' ')
            text = data_point
        elif isinstance(data_point, list):
            text = ' '.join(data_point).strip()
        else:
            raise ValueError('str | list of tokens input only')

        t = Text(
            0,
            data_point
        )
        corpus = Corpus(
            [t],
            add_subtexts=False,
            replace=False,
            subtext_types=False,
            sentence_fraction=0
        )
        dataset = CorefDataset(corpus, self.tokenizer)
        loader = CorefLoader(
            dataset, max_total_seq_len=self.cfg.model_config.max_total_seq_len, batch_size_1=True
        )

        batch = tuple(tensor.to(self.model.device) for tensor in loader[0][1])
        input_ids, attention_mask, gold_clusters = batch
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_all_outputs=True)

        pred = Evaluator.get_prediction(
            batch,
            outputs,
            tokenizer=self.tokenizer
        )

        api_text = api.Text(
            text=text,
            clusters=pred.clusters,
            singletons=pred.singletons,
            tokenized=pred.tokenized_text
        )
        aligned_text = align(api_text, data_point)
        if mention_heads:
            add_heads(aligned_text, self.nlp) # @TODO: turn off for Spacy-based inference

        return aligned_text
