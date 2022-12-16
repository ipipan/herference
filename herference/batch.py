import logging
from typing import List, Iterable
import math

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from herference.dataset import CorefDataset

logger = logging.getLogger(__name__)

max_model_input = 512


def calc_effective_per_example_batch_len(example_len):
    return 512 * math.ceil(
        (example_len + 2) / 512
    )


class CorefLoader(DataLoader):
    def __init__(
            self,
            data_source: CorefDataset,
            max_total_seq_len: int,
            sorting_keys: List[str] = None,
            padding_noise: float = 0.1,
            batch_size_1: bool = False
    ):
        super().__init__(
            data_source,
            num_workers=4,
            pin_memory=True
        )
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.max_total_seq_len = max_total_seq_len
        self.data_source = data_source
        data_source.tokenized_data_points.sort(key=lambda x: len(x[1].token_ids), reverse=True)
        self.batches = self.prepare_batches() if not batch_size_1 else self.prepare_eval_batches()
        self.batch_size_1 = batch_size_1

    def prepare_batches(self):
        batches = []
        batch = []
        per_data_point_batch_len = 0
        data_source_iterator = tqdm(self.data_source, desc="Iteration")
        for _, elem in data_source_iterator:
            logging.debug(f"elem no. {_} size: {len(elem)}")

            # if len(elem.token_ids) > max_model_input:
            #     logger.info(f"omit {_[0]} {len(elem.token_ids)} (prepare_batches)")
            #     continue

            if len(batch) == 0:
                per_data_point_batch_len = calc_effective_per_example_batch_len(len(elem.token_ids))

            elif (len(batch) + 1) * per_data_point_batch_len > self.max_total_seq_len:
                batch = self.data_source.pad_batch(batch, len(batch[0].token_ids))

                batches.append(batch)
                batch = []
                per_data_point_batch_len = calc_effective_per_example_batch_len(len(elem.token_ids))
            batch.append(elem)

        if len(batch) == 0:
            return batches
        batch = self.data_source.pad_batch(batch, len(batch[0].token_ids))
        batches.append(batch)
        return batches

    def prepare_eval_batches(self):
        batches = []
        for text_id, elem in self.data_source:
            # if len(elem.token_ids) > max_model_input:
            #     logger.info(f"omit {text_id[0]} {len(elem.token_ids)} (prepare_eval_batches)")
            #     continue
            batch = self.data_source.pad_batch([elem], len(elem.token_ids))
            batches.append(
                (text_id, batch)
            )
        return batches

    def __getitem__(self, item):
        return self.batches[item]

    def __iter__(self) -> Iterable[List[int]]:
        n_batches = len(self.batches)
        permutation = torch.randperm(n_batches)
        self.batches = [self.batches[ind] for ind in permutation]
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def print_batch(self, batch_ind, max_texts=3):
        batch = self.batches[batch_ind]
        if self.batch_size_1:
            batch = batch[1]
        texts, masks, clusters_s = batch
        for ind_text, text_token_ids in enumerate(texts):
            text_token_ids = text_token_ids.tolist()

            print(f"-----------\ntext no. {ind_text}")
            print(
                ''.join(
                    self.data_source.tokenizer.decode(text_token_ids)
                ),
                end=' | '
            )
            for cluster in clusters_s[ind_text]:
                print("cluster:")
                for start, end in cluster:
                    if start == end == 0:
                        print('-', end=' | ')
                        continue
                    print(
                        ''.join(
                            self.data_source.tokenizer.decode(text_token_ids[start.item():end.item() + 1])
                        ),
                        end=' | ')
                print('')
            if max_texts == ind_text:
                break
