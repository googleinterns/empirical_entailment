from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

def _step(model: Any
          ):

def generate(model: Any,
             input_ids: Tensor,
             max_length: int,
             min_length: Optional[int] = 0,
             num_beams: Optional[int] = None,
             no_repeat_ngram_size: Optional[int] = 0,
             ):

    assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    bos_token_id = model.config.bos_token_id
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id




    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1
    pass



class BeamHypotheses(object):
    """
    Taken from Facebook XLM code
    https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529
    """
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty


