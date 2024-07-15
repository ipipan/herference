from herference.api import (
    Text,
    Cluster,
    Mention
)


def is_mention_nested(a: Mention, b: Mention) -> bool:
    a_start, a_end = a.indices
    b_start, b_end = b.indices

    is_a_nested_in_b = a_start >= b_start and a_end <= b_end
    is_b_nested_in_a = a_start <= b_start and a_end >= b_end

    return is_a_nested_in_b or is_b_nested_in_a

def filtered_nested_mention_pairs_from_clusters(text: Text) -> list[Cluster]:
    filtered_clusters = []

    for cluster in text.clusters:
        if len(cluster) == 2:
            mention_a, mention_b = cluster
            if is_mention_nested(mention_a, mention_b):
                continue

        filtered_clusters.append(cluster)


    return filtered_clusters
