# Copyright (c) 2022-2023 CLARIN-PL Wroclaw University of Technology and Science
# All rights reserved.
#
#  This file incorporates work covered by the following copyright and
#  permission notice:
#    Copyright (c) 2021 (https://github.com/yuvalkirstain/s2e-coref/blob/main/eval.py)
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import json
import os
import logging
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from herference.api import Cluster, Mention
from herference.utils import (
    extract_clusters,
    extract_mentions_to_predicted_clusters_from_clusters,
    extract_clusters_for_decode,
    detokenize
)

from herference import api

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    mentions: List[Mention]
    clusters: List[Cluster]
    singletons: List[Tuple[int, int]]
    tokenized_text: List[str]
    singletons_meta: List[Tuple[int, int]] = None


class Evaluator:
    @staticmethod
    def get_prediction(batch, outputs, tokenizer):
        batch_np = tuple(tensor.detach().cpu().numpy() for tensor in batch)
        token_ids = batch_np[0]
        for k, v in outputs.items():
            outputs[k] = v.detach().cpu().numpy()
        for token_ids, starts, end_offsets, coref_logits, mention_reps, mention_logits in \
                zip(token_ids, outputs['mention_start_ids'], outputs['mention_end_ids'], outputs['coref_logits'],
                    outputs['mention_logits'], outputs['mention_reps']):

            mention_to_reps = {
                (s, e):  rep for s, e, rep in zip(starts, end_offsets, mention_reps)
            }

            max_antecedents = np.argmax(coref_logits, axis=1).tolist()
            mention_to_antecedent = {
                ((int(start), int(end)), (int(starts[max_antecedent]), int(end_offsets[max_antecedent])))
                # map mention start_ind, end_ind to antecedent start_ind, end_ind
                for start, end, max_antecedent in
                zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

            predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
            mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
            predicted_mentions = list(mention_to_predicted_clusters.keys())

            singletons = [
                [(int(start), int(end))]
                # mention start_ind, end_ind for singleton mentions
                for start, end, max_antecedent in
                zip(starts, end_offsets, max_antecedents) if
                max_antecedent == len(starts) and
                (int(start), int(end)) not in mention_to_predicted_clusters.keys()
            ]

            singletons_meta = [
                mention_logits[start, end]
                # mention start_ind, end_ind for singleton mentions
                for ind, (start, end, max_antecedent) in
                enumerate(zip(starts, end_offsets, max_antecedents)) if
                max_antecedent == len(starts) and
                (int(start), int(end)) not in mention_to_predicted_clusters.keys()
            ]
            singletons = sorted([(ind, men, singletons_meta[ind]) for ind, men in enumerate(singletons)],
                                key=lambda sing: singletons_meta[sing[0]])

            mentions = detokenize(tokenizer, token_ids, predicted_mentions)
            clusters = [
                api.Cluster(detokenize(tokenizer, token_ids, cluster)) for cluster in predicted_clusters]
            for cluster in clusters:
                for mention in cluster:
                    mention.vector = mention_to_reps[mention.indices]
            singletons = [(detokenize(tokenizer, token_ids, cluster[1]), cluster[2]) for cluster in singletons]
            for singleton in singletons:
                singleton[0][0].logit = singleton[1]
            singletons = [singleton[0][0] for singleton in singletons]

            tokenized_decoded = detokenize(tokenizer, token_ids, [(i, i) for i in range(len(token_ids))])
            return Prediction(
                mentions,
                clusters,
                singletons,
                [tok.text for tok in tokenized_decoded]
            )
