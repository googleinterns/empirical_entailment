from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import numpy as np

from .util import predict_rte_label

MODEL_DIR = f'models/modifier_entailment/'
# Label Set for the model: [entailment, not_entailment]
entailment_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
entailment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

entailment_model = entailment_model.cuda()
entailment_model.eval()


def calculate_entailment_score(text1_batch: List[str],
                               text2_batch: List[str]) -> np.ndarray:
    """
    Uses a pretrained entailment model to get the softmax score on the "entailment" label.

    :param text1_batch: Batch of sentence 1
    :param text2_batch: Batch of sentence 2
    :return:
    """
    return predict_rte_label(text1_batch, text2_batch, entailment_model, entailment_tokenizer)