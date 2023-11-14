import unittest
import spacy

from herference import *


def init_spacy(model: str):
    try:
        nlp = spacy.load(model)
    except IOError as e:
        spacy.cli.download(model)
        nlp = spacy.load(model)
    finally:
        nlp.add_pipe("herference")
        return nlp


class SpacyTests(unittest.TestCase):
    def test_predict(self):
        nlp = init_spacy("pl_core_news_sm")
        doc = nlp("Ala ma mruczącego kota, jest on bardzo ładny.")
        self.assertIsNotNone(doc._.coref.clusters[0].mentions[0].span)

    def test_load_specific_coref_model(self):
        model = "pl_core_news_sm"
        nlp = spacy.load(model)
        nlp.add_pipe("herference", config={"model_name_or_path": "ipipan/herference-base", "device": "cpu"})
        doc = nlp("Ala ma mruczącego kota.  Ala w ogóle lubi zwierzęta.")
        print('doc preds: ', doc._.coref)
        self.assertIsNotNone(doc._.coref.clusters[0].mentions[0].span)

    def test_alignment(self):
        nlp = init_spacy("pl_core_news_sm")
        from herference.tests.texts import texts
        for text in texts:
            doc = nlp(text)
            for c in doc._.coref:
                for m in c:
                    print(m, m.span is not None)
                    self.assertIsNotNone(m.span)


if __name__ == "__main__":
    unittest.main()
