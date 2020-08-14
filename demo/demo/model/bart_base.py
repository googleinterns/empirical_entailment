from .util import load_model_and_tokenizer_huggingface, produce_summary_huggingface
from typing import List, Tuple

_MODEL_DIR = f'models/bart_base/'
MODEL_NAME = 'bart.base'

model, tokenizer = load_model_and_tokenizer_huggingface(_MODEL_DIR)

model = model.cuda()
model.eval()

GENERATION_CONFIG = {
    "num_beams": 10,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "min_length": 10,
    "max_length": 60,
    "limit_vocab_to_input": False,
    "num_return_sequences": 10,
    "do_sample": True,
}


def produce_summary(source_text: str) -> List[Tuple[str, float]]:
    """
    Generates a short summary from the source text, using BART base model trained on XSUM dataset.
    :param source_text:
    :return: generated summary
    """
    return produce_summary_huggingface(source_text=source_text,
                                       model=model,
                                       tokenizer=tokenizer,
                                       config=GENERATION_CONFIG)
