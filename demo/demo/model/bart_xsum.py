import torch
from fairseq.models.bart import BARTModel

MODEL_NAME = 'bart.large.xsum'

bart = BARTModel.from_pretrained(
    '/scratch/sihaoc/models/bart.large.xsum/',
    checkpoint_file='model.pt',
)

bart.cuda()
bart.eval()
bart.half()

_config = {
    "beam": 6,
    "lenpen": 1.0,
    "max_len_b": 60,
    "min_len": 10,
    "no_repeat_ngram_size": 3
}


def produce_summary(source_text: str) -> str:
    """
    Generate a short summary from the source text
    :param source_text:
    :return: generated summary
    """
    source_b = [source_text.rstrip()]
    hypothesis_b = bart.sample(source_b, **_config)

    return hypothesis_b[0]
