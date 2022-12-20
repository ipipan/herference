from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Doc

from herference.manager import Herference
from herference.align import align

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


class SpacyCoref(Pipe):
    def __init__(self, nlp, name):
        self.nlp = nlp
        self.name = name
        self.cr = Herference()

    def __call__(self, doc):
        self.annotate(doc)
        return doc

    def annotate(self, doc):
        tokens = [str(tok) for tok in doc]
        text = self.cr.predict(tokens)
        aligned_text = align(text, doc)
        doc._.coref = aligned_text
        return doc
