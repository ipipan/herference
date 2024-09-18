
from transformers import AutoTokenizer

from herference.evaluator import Evaluator
from herference.align import align
from herference import api


def parse_serialized_pred(prediction_path, tokenizer=None):
    serialized_pred = api.SerializedPrediction(prediction_path)
    text = serialized_pred.text

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('ipipan/herference-large')

    pred = Evaluator.get_prediction(
        serialized_pred.get_batch(),
        serialized_pred.get_outputs(),
        tokenizer=tokenizer
    )  # @TODO refactor this function to not use a for-loop?

    api_text = api.Text(
        text=text,
        clusters=pred.clusters,
        singletons=pred.singletons,
        tokenized=pred.tokenized_text
    )
    aligned_text = align(api_text, text)

    # aligned_text.clusters = filtered_nested_mention_pairs_from_clusters(aligned_text)
    return aligned_text
