from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.tokens import Doc

from herference.manager import Herference
from herference.align import align

Doc.set_extension("coref", default=None)


@Language.factory(
    "herference",
    default_config={
        "model_name_or_path": None,
        "device": None,
        "cfg": None,
    }
)
def make_herference(nlp, name, model_name_or_path, device, cfg):
    return SpacyCoref(nlp, name, model_name_or_path, device, cfg)


class SpacyCoref(Pipe):
    def __init__(self, nlp, name, model_name_or_path, device, cfg):
        self.nlp = nlp
        self.name = name
        self.cr = Herference(_cfg=cfg, device=device, model=model_name_or_path)

    def __call__(self, doc):
        self.annotate(doc)
        return doc

    def annotate(self, doc):
        tokens = [str(tok) for tok in doc]
        text = self.cr.predict(tokens, mention_heads=False)
        aligned_text = align(text, doc)
        doc._.coref = aligned_text
        return doc
