import math

import torch

from herference.utils import get_sentences, get_span_len


def mask_tensor(t, mask):
    """
    Mask tensor t with large negative value.
    :param t:
    :param mask:
    :return:
    """
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t


def split_tokenized(t, max_length=510):
    t = t.squeeze()
    dot_token = 1899
    sentences = get_sentences(t, dot_token)
    n_sentences = len(sentences)
    spans = []

    curr_span_end = 0
    curr_span_start = 0

    for sent_ind, (start_ind, end_ind) in enumerate(sentences):
        sent_len = get_span_len(start_ind, end_ind)
        # if sentence can be included in the current span let's add it
        if get_span_len(curr_span_start, curr_span_end) + sent_len < max_length:
            curr_span_end = end_ind
        # if sentence can't be included in the current span
        else:
            # finish current span IF the span is not empty
            if get_span_len(curr_span_start, curr_span_end) > 0:
                spans.append(
                    (curr_span_start, curr_span_end)
                )
                curr_span_start = start_ind
                curr_span_end = None

            # if sentence is smaller/equal than max_length just start next span
            if sent_len <= max_length:
                curr_span_end = end_ind
            # if sentence is longer than max_length split the sentence
            else:
                n_splits = math.ceil(sent_len / max_length)
                # add full max_length splits as spans
                for split_number in range(1, n_splits):
                    curr_span_end = curr_span_start + (max_length - 1)
                    spans.append(
                        (curr_span_start, curr_span_end)
                    )
                    curr_span_start = curr_span_end + 1
                    curr_span_end = None
                # initialize new span for the rest of sentence
                curr_span_end = curr_span_start + (max_length - 1)
                # which will be added to the next span

        # if it is last element in the loop end the span
        if sent_ind == n_sentences - 1:
            curr_span_end = end_ind
            spans.append(
                (curr_span_start, curr_span_end)
            )

    list_of_sizes = [get_span_len(start, end) for start, end in spans]
    return torch.split(t, list_of_sizes), spans
