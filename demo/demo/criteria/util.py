from typing import List
import torch
import numpy as np

import torch
from transformers import glue_convert_examples_to_features, InputExample, PreTrainedModel, PreTrainedTokenizer

F_SOFTMAX = torch.nn.Softmax(dim=1)


def predict_rte_label(text1_batch: List[str],
                      text2_batch: List[str],
                      entailment_model: PreTrainedModel,
                      entailment_tokenizer: PreTrainedTokenizer) -> np.ndarray:
    entailment_logits = _predict_batch(text1_batch, text2_batch, entailment_model, entailment_tokenizer)
    prob = F_SOFTMAX(entailment_logits.logits)
    entailment_prob = prob[:, 0]
    entailment_prob = entailment_prob.cpu().numpy()
    return entailment_prob


def predict_mnli_label(text1_batch: List[str],
                       text2_batch: List[str],
                       entailment_model: PreTrainedModel,
                       entailment_tokenizer: PreTrainedTokenizer) -> np.ndarray:
    entailment_logits = _predict_batch(text1_batch, text2_batch, entailment_model, entailment_tokenizer)
    prob = F_SOFTMAX(entailment_logits.logits)
    entailment_prob = prob[:, 2]
    entailment_prob = entailment_prob.cpu().numpy()
    return entailment_prob


def _predict_batch(text1_batch: List[str],
                   text2_batch: List[str],
                   entailment_model: PreTrainedModel,
                   entailment_tokenizer: PreTrainedTokenizer):
    """
    Run a sentence pair to a pretrained entailment model

    :param text1_batch: Batch of sentence 1
    :param text2_batch: Batch of sentence 2
    :param entailment_model:
    :param entailment_tokenizer:
    :return:
    """
    assert len(text1_batch) == len(text2_batch), "Batch size of two inputs should be equal!"

    entailment_input = [InputExample(text_a=text1,
                                     text_b=text2_batch[idx],
                                     guid="") for idx, text1 in enumerate(text1_batch)]

    entailment_features = glue_convert_examples_to_features(entailment_input,
                                                            tokenizer=entailment_tokenizer,
                                                            label_list=['contradiction', 'neutral', 'entailment'],
                                                            output_mode="classification")

    all_input_ids = torch.tensor([f.input_ids for f in entailment_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in entailment_features], dtype=torch.long)

    all_input_ids = all_input_ids.to('cuda')
    all_attention_mask = all_attention_mask.to('cuda')

    with torch.no_grad():
        entailment_output = entailment_model(input_ids=all_input_ids,
                                             attention_mask=all_attention_mask)

    return entailment_output