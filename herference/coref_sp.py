from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.tokens import Span
import spacy_alignments

from herference import api
from herference.manager import Herference

Doc.set_extension("coref", default=None)


@Language.factory(
    "herference",
    default_config={
        "model": None,
        "mode": "lookup",
        "overwrite": False,
        "scorer": {"@scorers": "spacy.lemmatizer_scorer.v1"}
    }
)
def make_herference(nlp, model, name, mode, overwrite, scorer):
    return SpacyCoref(nlp, name)


def intersect(span: Span, text: str):
    span_toks = {str(tok) for tok in span}
    text_toks = {tok.strip('",:.?!') for tok in text.split(' ')}

    return any(text_toks.intersection(span_toks))


class SpacyCoref(Pipe):
    def __init__(self, nlp, name):
        self.nlp = nlp
        self.name = name
        self.cr = Herference()

    def __call__(self, doc):
        self.annotate(doc)
        return doc

    def align_mention(self, doc, mention, subtoken2token_indices):
        start, end = mention.inds
        if subtoken2token_indices[start] and subtoken2token_indices[end]:
            start, end = (
                subtoken2token_indices[start][0],
                subtoken2token_indices[end][0]
            )
            span = doc[start:(end + 1)]
            if intersect(span, mention.text):
                mention.span = span
            # else:
            #     print(span, mention.text)
        return mention

    def align(self, text: api.Text, doc: Doc):
        a2b, b2a = spacy_alignments.get_alignments(text.tokenized, [str(tok) for tok in doc])
        for cluster in text.clusters:
            for mention in cluster.mentions:
                self.align_mention(doc, mention, a2b)

        for singleton in text.singletons:
            self.align_mention(doc, singleton, a2b)

        return text

    def annotate(self, doc):
        tokens = [str(tok) for tok in doc]
        text = self.cr.predict(tokens)
        aligned_text = self.align(text, doc)
        doc._.coref = aligned_text
        return doc
