from .util import produce_summary_np_constrained_decoding
from typing import List, Tuple
from .bart_base import model, tokenizer

MODEL_NAME = 'bart.base.np.constrained.decoding'

GENERATION_CONFIG = {
    "num_beams": 5,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "min_length": 10,
    "max_length": 30,
    "temperature": 1000,
    "num_return_sequences": 3,
    "limit_vocab_to_input": False,
    "do_sample": False,
}


def produce_summary(source_text: str) -> List[Tuple[str, float]]:
    """
    Generates a short summary from the source text, using BART base model trained on XSUM dataset.
    The decoder will favor noun phrases appeared in the orignal text during the decoding process.
    :param source_text:
    :return: generated summary
    """
    return produce_summary_np_constrained_decoding(source_text=source_text,
                                                   model=model,
                                                   tokenizer=tokenizer,
                                                   config=GENERATION_CONFIG)
