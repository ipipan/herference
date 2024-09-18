import math


from herference import api


def flatten_list_of_lists(lst):
    return [elem for sub_lst in lst for elem in sub_lst]


def extract_clusters(clusters, null_id_for_coref=0):
    clusters = [
        tuple(tuple(mention)
              for mention in cluster if null_id_for_coref not in mention)
        for cluster in clusters.tolist()
    ]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    return clusters


def extract_mentions_to_predicted_clusters_from_clusters(clusters):
    mention_to_cluster = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster[tuple(mention)] = cluster
    return mention_to_cluster


def get_span_len(start, end):
    if not end:
        return 0
    else:
        return end - (start - 1)


def get_sentences(t, dot_token):
    text_len = t.shape[0]
    last_ind = text_len - 1
    sentences = []
    start_ind = 0
    for end_ind in range(text_len):
        if end_ind == last_ind - 1:  # end of text with end text token
            end_ind += 1  # include end of text token
            sentences.append(
                (start_ind, end_ind)
            )
            break  # don't consider last token separately
        elif t[end_ind] == dot_token:
            sentences.append(
                (start_ind, end_ind)
            )
            start_ind = end_ind + 1

    return sentences


def extract_clusters_for_decode(mention_to_antecedent):
    mention_to_antecedent = sorted(mention_to_antecedent)
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            clusters[cluster_idx].append(mention)
            mention_to_cluster[mention] = cluster_idx

        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster


def detokenize(tokenizer, tensor, ranges=None, starts=None, ends=None):
    if starts is None and ends is None:
        starts = [0]
        ends = [tensor.shape[0]]

    if ranges is None:
        ranges = zip(starts, ends)
    elif len(ranges) == 0:  # empty cluster passed
        return list()
    return [
        api.Mention(
            text=tokenizer.decode(
                tensor[start:end+1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ),
            indices=(start, end)
        )
        for start, end in ranges
    ]
