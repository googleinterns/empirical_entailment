import torch
from .util import produce_summary_fairseq

MODEL_NAME = 'bart.large.cnn'

bart = torch.hub.load('pytorch/fairseq', MODEL_NAME)

bart.cuda()
bart.eval()
bart.half()

GENERATION_CONFIG = {
    "beam": 4,
    "lenpen": 2.0,
    "max_len_b": 140,
    "min_len": 55,
    "no_repeat_ngram_size": 3
}


def produce_summary(source_text: str) -> str:
    """
    Generates a short summary from the source text, using BART large model finetuned on the CNN/Daily Mail dataset.
    :param source_text:
    :return: generated summary
    """
    return produce_summary_fairseq(source_text=source_text,
                                   model=bart,
                                   config=GENERATION_CONFIG)
