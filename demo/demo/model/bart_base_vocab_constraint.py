import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_MODEL_DIR = f'models/bart_base/'

MODEL_NAME = 'bart.base.vocab.input'

model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)

model = model.cuda()
model.eval()

GENERATION_CONFIG = {
    "num_beams": 6,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "min_length": 10,
    "max_length": 60,
    "limit_vocab_to_input": True,
    "do_sample": True,
}


def produce_summary(source_text: str) -> str:
    """
    Generate a short summary from the source text
    :param source_text:
    :return: generated summary
    """
    input_ids = torch.tensor(tokenizer.encode(source_text, add_special_tokens=True)).unsqueeze(0)
    input_ids = input_ids.to('cuda')
    generated = model.generate(input_ids, **GENERATION_CONFIG)
    gen_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    gen_text = gen_text.strip()
    return gen_text


if __name__ == '__main__':
    print(produce_summary("Brooks Brothers, the clothier that traces its roots to 1818, filed for bankruptcy. Harvard and M.I.T. sued the Trump administration over its plan to require foreign students to attend classes in person."))