from __future__ import annotations

import logging
from typing import List

import spacy_alignments
from spacy.tokens import Doc

from herference import api


logger = logging.getLogger(__name__)


def intersect(span, text: list[str]):
    span_toks = {str(tok) for tok in span}
    text_toks = {tok.strip('",:.?!') for tok in text}

    return any(text_toks.intersection(span_toks))


def align_mention(doc: Doc | List[str], mention: api.Mention, subtoken2token_indices) -> api.Mention:
    start, end = mention.subtoken_indices if isinstance(doc, Doc) else mention.indices

    if subtoken2token_indices[start] and subtoken2token_indices[end]:
        start, end = (
            subtoken2token_indices[start][0],
            subtoken2token_indices[end][0]
        )
        span = doc[start:(end + 1)]
        if isinstance(doc, Doc) and intersect(span, mention.text):
            mention.span = span
        elif isinstance(doc, list):
            mention.indices = (start, end)
            mention.text = span
        else:
            logger.debug(f"{span}, {mention.text}")
    return mention


def align(text: api.Text, doc: Doc | List[str]) -> api.Text:
    a2b, b2a = spacy_alignments.get_alignments(text.tokenized, [str(tok) for tok in doc])
    for cluster in text.clusters:
        for mention in cluster.mentions:
            align_mention(doc, mention, a2b)
    if text.singletons:
        for singleton in text.singletons:
            align_mention(doc, singleton, a2b)

    return text
