import unittest
import spacy
import herference


def init_spacy(model: str):
    try:
        nlp = spacy.load(model)
    except IOError as e:
        spacy.cli.download(model)
        nlp = spacy.load(model)
    finally:
        nlp.add_pipe('herference')
        return nlp


class MyTestCase(unittest.TestCase):
    def test_predict(self):
        nlp = init_spacy("pl_core_news_sm")
        doc = nlp("Ala ma mruczącego kota, jest on bardzo ładny.")
        self.assertIsNotNone(doc._.coref.clusters[0].mentions[0].span)


if __name__ == '__main__':
    unittest.main()
