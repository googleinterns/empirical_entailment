from .util import produce_summary_huggingface_custom_decoding
from typing import List, Tuple
from .bart_base import model, tokenizer

MODEL_NAME = 'bart.base.upweight.np'

GENERATION_CONFIG = {
    "num_beams": 5,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "min_length": 10,
    "max_length": 60,
    "limit_vocab_to_input": False,
    "num_return_sequences": 5,
    "do_sample": True,
}


def produce_summary(source_text: str) -> List[Tuple[str, float]]:
    """
    Generates a short summary from the source text, using BART base model trained on XSUM dataset.
    :param source_text:
    :return: generated summary
    """
    return produce_summary_huggingface_custom_decoding(source_text=source_text,
                                                       model=model,
                                                       tokenizer=tokenizer,
                                                       config=GENERATION_CONFIG)
