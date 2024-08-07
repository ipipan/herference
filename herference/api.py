from dataclasses import dataclass
from typing import Tuple, List

from spacy.tokens import Span


@dataclass
class Mention:
    """
    Mention object - a reference to a text span
    :param text: string with text of the mention
    :param indices: tuple of start and end indices of the mention
    :param tokenized_indices: indices of mention range in Text.tokenized
    :param span: Spacy span of a mention if it was found
    :param logit: Logit value of significance - used in case of singleton mentions
    """
    text: List[str]
    indices: Tuple[int, int]
    subtoken_indices: Tuple[int, int] = None
    span: Span = None
    logit: float = None
    head: int = None

    def __repr__(self):
        if self.span:
            return f'{self.span}'
        else:
            return f'{self.text} <{self.indices[0]}, {self.indices[1]}>'
        

    def __iter__(self):
        for el in self.indices:
            yield el

    def __post_init__(self):
        self.subtoken_indices = self.indices


@dataclass
class Cluster:
    mentions: List[Mention]

    def __repr__(self):
        return ' | '.join(
            [mention.__repr__() for mention in self.mentions]
        )

    def __getitem__(self, item):
        return self.mentions[item]

    def __len__(self):
        return len(self.mentions)

    def __iter__(self):
        return iter(self.mentions)


@dataclass
class Text:
    """

    :param text: string with text
    :param clusters: list of clusters detected in text
    :param singletons: list of
    mentions detected in text. It is sorted list with .logit value of possible significant non-coreferent mentions
    :param tokenized: list of tokens of text
    """
    text: str
    clusters: List[Cluster]
    singletons: List[Mention] = None
    tokenized: List[str] = None

    @property
    def mentions(self):
        for cluster in self.clusters:
            for mention in cluster:
                yield mention

    def __repr__(self):
        return ''.join(
            [f'\n\n --- Cluster {ind} --- \n' + cluster.__repr__() for ind, cluster in enumerate(self.clusters)]
        ).strip()

    def __getitem__(self, item):
        return self.clusters[item]

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        return iter(self.clusters)
