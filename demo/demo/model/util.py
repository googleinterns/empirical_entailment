from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, Tuple, List
from fairseq.models import BaseFairseqModel
from .entailment_model import calculate_entailment_score
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
                               config: Optional[dict] = None) -> List[Tuple[str, float]]:
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
    gen_text_batch = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    source_text_batch = [source_text] * len(gen_text_batch)
    entailment_score = calculate_entailment_score(source_text_batch, gen_text_batch)

    entailment_score = entailment_score.tolist()
    res = list(zip(gen_text_batch, entailment_score))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res


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
    TODO: need to connect entailment model to this
    :param source_text: Source text/paragraph to produce summary on
    :param model: Pretrained fairseq model
    :param config: Configuration for generation/decoding.
    :return: Produced summary by the model
    """
    source_b = [source_text.rstrip()]
    hypothesis_b = model.sample(source_b, **config)

    return hypothesis_b[0]