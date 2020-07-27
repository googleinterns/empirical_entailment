import os
import torch
import numpy as np
import tqdm

from typing import Optional, List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import InputExample, glue_convert_examples_to_features
from transformers import PreTrainedTokenizer, PreTrainedModel

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

from scipy.stats import spearmanr

def produce_summary(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    source_text: str,
                    cuda: bool,
                    generation_configs: Optional[dict] = None) -> List[str]:
    """
    Use a summarization model to generate output
    :param model: Pretrained Summarization model
    :param tokenizer: Tokenizer for the summarization model
    :param source_text:
    :param cuda:
    :return:
    """
    input_ids = torch.tensor(tokenizer.encode(source_text, truncation=True, add_special_tokens=True)).unsqueeze(0)

    if cuda:
        input_ids = input_ids.to('cuda')

    generated = model.generate(input_ids, **generation_configs)
    gen_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return gen_text


entailment_label_set = ['C', 'N', 'E']
F_SOFTMAX = torch.nn.Softmax(dim=1)


def get_entailment_prob(model,
                        tokenizer,
                        premise: str,
                        hypo: List[str],
                        cuda: bool) -> str:
    """

    :param model: Entailment model
    :param tokenizer:
    :param premise: Premise
    :param hypo: Hypothesis
    :param cuda: True is cuda is used
    :return: softmax probability output of "entailment" label
    """
    entailment_input = [InputExample(text_a=premise, text_b=_hypo, guid="") for _hypo in hypo]
    entailment_features = glue_convert_examples_to_features(entailment_input,
                                                            tokenizer=tokenizer,
                                                            label_list=['contradiction', 'neutral', 'entailment'],
                                                            output_mode="classification")
    all_input_ids = torch.tensor([f.input_ids for f in entailment_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in entailment_features], dtype=torch.long)

    if cuda:
        all_input_ids = all_input_ids.to('cuda')
        all_attention_mask = all_attention_mask.to('cuda')

    with torch.no_grad():
        entailment_logits = model(input_ids=all_input_ids,
                                  attention_mask=all_attention_mask)

        _softmax = entailment_logits[0]
        entailment_softmax = _softmax[:, 2]
        entailment_softmax = entailment_softmax.cpu().numpy()

    return entailment_softmax


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


def evaluate_entailment(data_dir: str,
                        model_dir: str,
                        cuda: bool = True,
                        limit: int = 10) -> None:
    """
    Use a pretrained entailment model to evaluate the output of a summarization model against
    (1) source text
    (2) reference summary

    :param data_dir: Directory containing XSum or other dataset (in huggingface format,
    see https://github.com/huggingface/transformers/tree/master/examples/seq2seq for more details)
    :param model_dir: Directory containing saved check point for BERT-like model.
    :param cuda: True if cuda is used
    :param limit: Max number of examples  used for this evaluation
    :return:
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    entailment_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
    entailment_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')

    if cuda:
        model = model.cuda()
        entailment_model.cuda()

    source_file = os.path.join(data_dir, "val.source")
    target_file = os.path.join(data_dir, "val.target")

    with open(source_file) as fin:
        source_lines = fin.readlines()

    with open(target_file) as fin:
        target_lines = fin.readlines()

    entail_gold = 0
    entail_source = 0
    total = 0

    entailment_source_score_list = []
    entailment_target_score_list = []

    rouge_score_list = []

    num_examples = min(len(source_lines), limit)
    for idx in tqdm.trange(num_examples):
        sl = source_lines[idx]
        tl = target_lines[idx]
        pred_summary = produce_summary(model, tokenizer, sl, cuda, generation_configs=GENERATION_CONFIG)
        entailment_source_score = get_entailment_prob(entailment_model, entailment_tokenizer, sl, pred_summary, cuda)
        entailment_target_score = get_entailment_prob(entailment_model, entailment_tokenizer, tl, pred_summary, cuda)

        rouge_s = np.array([scorer.score(tl, pred)['rougeL'].fmeasure for pred in pred_summary])

        entailment_source_score_list.append(entailment_source_score)
        entailment_target_score_list.append(entailment_target_score)

        rouge_score_list.append(rouge_s)

        total += 1

    entailment_source_score_list = np.concatenate(entailment_source_score_list)
    entailment_target_score_list = np.concatenate(entailment_target_score_list)
    rouge_score_list = np.concatenate(rouge_score_list)

    source_r = spearmanr(entailment_source_score_list, rouge_score_list)
    target_r = spearmanr(entailment_target_score_list, rouge_score_list)

    print("Total :\t{}".format(total))
    print("Spearman Corr between entailment (source) and rouge :\t{}".format(source_r))
    print("Spearman Corr between entailment (target) and rouge :\t{}".format(target_r))


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python ... [xsum_huggingface_dir] [model_dir]", file=sys.stderr)
        exit(1)

    xsum_huggingface_dir = sys.argv[1]
    _model_dir = sys.argv[2]

    evaluate_entailment(xsum_huggingface_dir, _model_dir)
