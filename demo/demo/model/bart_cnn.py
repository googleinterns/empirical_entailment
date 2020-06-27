import torch
from fairseq.models.bart import BARTModel

bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')


bart.cuda()
bart.eval()
bart.half()

_config = {
    "beam": 4,
    "lenpen": 2.0,
    "max_len_b": 140,
    "min_len": 55,
    "no_repeat_ngram_size": 3
}


def produce_summary(source_text):
    source_b = [source_text.rstrip()]
    hypothesis_b = bart.sample(source_b, **_config)

    return hypothesis_b[0]
