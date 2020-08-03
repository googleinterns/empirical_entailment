from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, Tuple, Any
from fairseq.models import BaseFairseqModel
import torch


DEFAULT_CONFIG = {
    "num_beams": 6,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "min_length": 10,
    "max_length": 60,
    "do_sample": True,
}


def produce_summary_huggingface(source_text: str,
                               model: PreTrainedModel,
                               tokenizer: PreTrainedTokenizer,
                               config: Optional[dict] = None) -> str:
    """
    Generates a summary using a model (huggingface's transformers library implementation).
    :param source_text: Source text/paragraph to produce summary on
    :param model: Pretrained transformer model
    :param tokenizer: corresponding tokenizer used for the model
    :param config: Configuration for generation/decoding.
    :return: Produced summary by the model
    """

    if not config:
        config = DEFAULT_CONFIG

    input_ids = torch.tensor(tokenizer.encode(source_text, add_special_tokens=True)).unsqueeze(0)
    input_ids = input_ids.to('cuda')
    generated = model.generate(input_ids, **config)
    gen_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    gen_text = gen_text.strip()
    return gen_text


def load_model_and_tokenizer_huggingface(model_name_or_dir: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Downloads/Loads a pretrained huggingface model and tokenizer.
    :param model_name_or_dir: Directory or the name of the model. See https://huggingface.co/ for the complete list
    :return: Pretrained model + tokenizer instance
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    return model, tokenizer


def produce_summary_fairseq(source_text: str,
                            model: BaseFairseqModel,
                            config: dict) -> str:
    """
    Generates a summary with a pretrained fairseq model.
    :param source_text: Source text/paragraph to produce summary on
    :param model: Pretrained fairseq model
    :param config: Configuration for generation/decoding.
    :return: Produced summary by the model
    """
    source_b = [source_text.rstrip()]
    hypothesis_b = model.sample(source_b, **config)

    return hypothesis_b[0]