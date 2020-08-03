import torch
from .util import produce_summary_fairseq

MODEL_NAME = 'bart.large.xsum'

bart = torch.hub.load('pytorch/fairseq', 'bart.large.xsum')

bart.cuda()
bart.eval()
bart.half()

GENERATION_CONFIG = {
    "beam": 6,
    "lenpen": 1.0,
    "max_len_b": 60,
    "min_len": 10,
    "no_repeat_ngram_size": 3
}


def produce_summary(source_text: str) -> str:
    """
    Generates a short summary from the source text, using BART large model finetuned on the XSUM dataset.
    :param source_text:
    :return: generated summary
    """
    return produce_summary_fairseq(source_text=source_text,
                                   model=bart,
                                   config=GENERATION_CONFIG)
