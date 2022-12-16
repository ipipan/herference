from __future__ import annotations
import logging
import copy
from pathlib import Path
from typing import List
from collections import namedtuple

import torch
from torch.utils.data import (
    Dataset
)
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm

from herference.utils import flatten_list_of_lists
from herference.config import get_config

CorefDataPoint = namedtuple("CorefDataPoint", ["token_ids", "clusters"])
TextTuple = namedtuple("DataPoint", ["text_id", "segments", "clusters"])

NULL_ID_FOR_COREF = get_config().model_config.null_id_for_coref
logger = logging.getLogger(__name__)


class Text:
    def __init__(
            self,
            text_id: int,
            segments: List[str],
            segments_meta: List[dict] = None,
            clusters: List[List[int]] = None
    ):
        self.text_id = text_id
        self.segments = segments
        self.clusters = clusters if clusters else []
        self.segments_meta = segments_meta

    @classmethod
    def text_with_merged_words(cls,
                               text_id: int,
                               segments: List[str],
                               segments_meta: List[dict] = None,
                               clusters: List[List[int]] = None):
        text = cls(
            text_id,
            segments,
            segments_meta,
            clusters
        )
        if segments_meta:
            text = cls.merge_split_words(text)
        return text

    def print_mentions(self, verbose=False):
        mentions = [el for cluster in self.clusters for el in cluster]
        mentions_str = [' '.join(self.segments[start:end + 1]) for start, end in mentions]
        if verbose:
            print(self.text_id, mentions_str)
        return mentions_str

    def print_clusters(self, verbose=False):
        clusters = []
        for cluster in self.clusters:
            mentions_str = [f"{' '.join(self.segments[start:end + 1])} ({start}, {end})" for start, end in cluster]
            clusters.append(mentions_str)
        if verbose:
            print(self.text_id, clusters)
        return clusters

    @staticmethod
    def update_indexes(clusters, token_ind):
        for cluster in clusters:
            for ind, (start, end) in enumerate(cluster):
                if end > token_ind:
                    end -= 1
                if start > token_ind:
                    start -= 1
                cluster[ind] = start, end
        return clusters

    @staticmethod
    def is_to_merge(text, ind):
        c_tags = ['interp', 'ign']
        return text.segments_meta[ind + 1]['ctag'] not in c_tags \
            and text.segments_meta[ind]['ctag'] not in c_tags \
            and text.segments_meta[ind + 1]['hasnps'] is True

    @staticmethod
    def merge_split_words(text, verbose=False):
        text = copy.deepcopy(text)
        merged_segments = []
        merged_segments_meta = []
        omit_next = False
        last_ind = len(text.segments) - 1
        penultimate_ind = last_ind - 1

        for ind in range(last_ind):
            if omit_next:
                omit_next = False
                continue
            elif Text.is_to_merge(text, ind):
                omit_next = True
                merged_segments.append(
                    text.segments[ind] + text.segments[ind + 1]
                )
                if verbose:
                    print(text.segments[ind] + text.segments[ind + 1], text.segments[ind], text.segments[ind + 1],
                          text.segments_meta[ind]['ctag'], text.segments_meta[ind + 1]['ctag'])
                curr_new_ind = len(merged_segments) - 1
                text.clusters = Text.update_indexes(text.clusters, curr_new_ind)
                text.segments_meta[ind].update(
                    text.segments_meta[ind + 1]
                )
                text.segments_meta[ind]['hasnps'] = False
                merged_segments_meta.append(
                    text.segments_meta[ind]
                )
                if ind == penultimate_ind - 1:
                    merged_segments.append(text.segments[last_ind])
                    merged_segments_meta.append(text.segments_meta[last_ind])
            elif ind == penultimate_ind:
                merged_segments.extend(text.segments[ind:])
                merged_segments_meta.extend(
                    text.segments_meta[ind:]
                )
            else:
                merged_segments.append(text.segments[ind])
                merged_segments_meta.append(
                    text.segments_meta[ind]
                )
        text.segments = merged_segments
        text.segments_meta = merged_segments_meta
        return text

    @staticmethod
    def trim_indexes_after_split(text, text_start_ind, text_end_ind):
        text_range = range(text_start_ind, text_end_ind)
        new_clusters = []
        for cluster_ind, cluster in enumerate(text.clusters):
            new_spans = []
            for span_start, span_end in cluster:
                if span_start not in text_range or span_end not in text_range:
                    continue
                else:
                    new_span = span_start - text_start_ind, span_end - text_start_ind
                    new_spans.append(new_span)

            if new_spans:
                new_clusters.append(new_spans)
        text.clusters = new_clusters
        return text

    @staticmethod
    def get_subtexts(text, split_key='lastinsent'):
        subtexts = []
        curr_subtext = copy.deepcopy(text)
        curr_segments = []
        curr_start_ind = 0

        name_dict = {'lastinsent': 's', 'lastinpar': 'p'}

        for ind, segment in enumerate(text.segments):
            curr_segments.append(segment)

            if text.segments_meta[ind][split_key]:
                curr_subtext.segments = curr_segments
                curr_subtext.text_id += f"_{name_dict[split_key]}_{len(subtexts)}"
                Text.trim_indexes_after_split(curr_subtext, curr_start_ind, ind)
                subtexts.append(curr_subtext)
                curr_subtext = copy.deepcopy(text)
                curr_segments = []
                curr_start_ind = ind + 1

        return subtexts


class Corpus:
    def __init__(self, texts: List[Text], add_subtexts, replace, subtext_types, sentence_fraction):
        self.texts = texts
        if not subtext_types:
            subtext_types = ['lastinpar', 'lastinsent']
        if add_subtexts:
            subtexts = []
            for subtext_type in subtext_types:
                cur_fraction = sentence_fraction if subtext_type == 'lastinsent' else 1
                subtexts.extend(
                    Corpus.subtexts_subset(self.get_subtexts(subtext_type), cur_fraction)
                )
            if replace:
                self.texts = subtexts
            else:
                self.texts.extend(subtexts)

    @staticmethod
    def subtexts_subset(subtexts: List[Text], fraction) -> List[Text]:
        n_texts = len(subtexts)
        permutation = torch.randperm(n_texts)
        n_fraction = int(fraction * n_texts)

        return [
            subtexts[permutation[ind]] for ind in range(n_fraction)
        ]

    # @classmethod
    # def load_from_tei(cls,
    #                   corpus_path,
    #                   cfg_format
    #                   ) -> Corpus:
    #     corpus_path = Path(corpus_path)
    #     corpus_iterator = tqdm(corpus_path.iterdir(), desc="Iteration")
    #     texts = [
    #         Corpus.convert_tei_to_corpus_text(
    #             tei.read(text_path, add_single_mentions_to_cluster=cfg_format.add_single_mentions_as_clusters)
    #         ) for text_path in corpus_iterator
    #     ]
    #
    #     return cls(texts,
    #                add_subtexts=cfg_format.add_subtexts,
    #                replace=cfg_format.remove_original_texts,
    #                subtext_types=cfg_format.subtext_types,
    #                sentence_fraction=cfg_format.fraction_of_sentences_to_add
    #                )

    # @staticmethod
    # def convert_tei_to_corpus_text(tei_text: tei.Text) -> Text:
    #     return Text.text_with_merged_words(
    #         tei_text.text_id,
    #         tei_text.segments,
    #         tei_text.segments_dicts,
    #         tei_text.clusters
    #     )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index) -> Text:
        return self.texts[index]

    def get_subtexts(self, split_key):
        new_texts = []
        for text in self.texts:
            new_texts.extend(
                Text.get_subtexts(text, split_key)
            )
        return new_texts

    def print_texts(self, ind_range):
        for ind in ind_range:
            text = self.texts[ind]
            print(f"{text.text_id}: {' '.join(text.segments)}\n{text.print_mentions()}")


class CorefDataset(Dataset):
    def __init__(
            self,
            corpus: Corpus,
            tokenizer: AutoTokenizer,
            max_seq_len=-1
    ):
        self.data_points, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = \
            CorefDataset.parse_data_points(corpus)
        self.max_seq_len = max_seq_len
        logger.info("Load tokenizer")
        self.tokenizer = tokenizer
        logger.info("Tokenization")
        self.tokenized_data_points, self.lengths, self.num_examples_filtered = self._tokenize(self.data_points)
        logger.info("Finished tokenization")

    def get_main_field(self):
        return self.tokenized_data_points

    def __len__(self):
        return len(self.get_main_field())

    def __getitem__(self, index) -> T_co:
        return self.get_main_field()[index]

    @staticmethod
    def parse_data_points(corpus: Corpus):
        data_points = []
        max_mention_num = float('-inf')
        max_cluster_size = float('-inf')
        max_num_clusters = float('-inf')

        for text in corpus:
            curr_mention_num = len(flatten_list_of_lists(text.clusters))
            curr_num_clusters = len(text.clusters) if text.clusters else 0
            curr_text_max_cluster_size = max(len(cluster) for cluster in text.clusters) if text.clusters else 0

            max_mention_num = max(max_mention_num, curr_mention_num)
            max_cluster_size = max(max_cluster_size, curr_text_max_cluster_size)
            max_num_clusters = max(max_num_clusters, curr_num_clusters)

            data_points.append(
                TextTuple(
                    text.text_id,
                    text.segments,
                    text.clusters
                )
            )

        return data_points, max_mention_num, max_cluster_size, max_num_clusters

    def _tokenize(self, data_points):
        tokenized_data_points = []
        lengths = []
        num_examples_filtered = 0
        for doc_key, words, clusters in data_points:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = [0]  # for <s>

            token_ids = []
            for idx, word in enumerate(words):
                word_idx_to_start_token_idx[idx] = len(token_ids) + 1  # +1 for <s>
                tokenized = self.tokenizer.encode(
                    [word],
                    add_special_tokens=False,
                    is_split_into_words=True
                )
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)
                word_idx_to_end_token_idx[idx] = len(token_ids)  # old_seq_len + 1 (for <s>) + len(tokenized_word) -
                # 1 (we start counting from zero) = len(token_ids)

            if 0 < self.max_seq_len < len(token_ids):
                num_examples_filtered += 1
                continue
            try:
                new_clusters = [
                    [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                    cluster in clusters
                ]
                lengths.append(len(token_ids))
            except Exception as e:
                logger.error(doc_key)
                logger.debug(word_idx_to_start_token_idx)
                logger.debug(word_idx_to_end_token_idx)
                logger.debug(clusters)
                raise e

            tokenized_data_points.append(
                (
                    (doc_key, end_token_idx_to_word_idx),
                    CorefDataPoint(token_ids=token_ids, clusters=new_clusters)
                )
            )
        return tokenized_data_points, lengths, num_examples_filtered

    def pad_clusters_inside(self, clusters):
        return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_cluster_size - len(cluster)) for cluster
                in clusters]

    def pad_clusters_outside(self, clusters):
        return clusters + [[]] * (self.max_num_clusters - len(clusters))

    def pad_clusters(self, clusters):
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def pad_batch(
            self,
            batch: List[CorefDataPoint],
            max_length: int
    ):
        max_length += 2  # we have additional two special tokens <s>, </s>
        padded_batch = []
        for data_point in batch:
            # noinspection PyCallingNonCallable
            encoded_dict = self.tokenizer.prepare_for_model(data_point.token_ids,
                                                            add_special_tokens=True,
                                                            is_split_into_words=True,
                                                            padding='max_length',
                                                            max_length=max_length,
                                                            truncation=False,
                                                            return_attention_mask=True,
                                                            return_tensors='pt')
            clusters = self.pad_clusters(data_point.clusters)
            data_point_ = (encoded_dict["input_ids"], encoded_dict["attention_mask"]) + (torch.tensor(clusters),)
            padded_batch.append(data_point_)

        max_len = len(data_point_)
        tensored_batch = tuple(
            torch.stack(
                [data_point_[i].squeeze() for data_point_ in padded_batch], dim=0
            ) for i in range(max_len)
        )

        return tensored_batch
