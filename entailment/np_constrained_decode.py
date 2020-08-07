import logging
from typing import Any, Optional, List, Union, Iterable, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer
import spacy

import torch
from torch import Tensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


def get_tokenized_noun_phrases(source_text: str,
                               tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    doc = nlp(source_text)
    noun_phrases = []
    # for _np in doc.noun_chunks:
    #     _toks = []
    #     for i in range(_np.start, _np.end):
    #         _toks.append(doc[i].text)
    #     _text = " ".join(_toks)
    #     noun_phrases.append(tokenizer.encode(_text, add_special_tokens=False))

    for _np in doc.ents:
        noun_phrases.append(tokenizer.encode(_np.text, add_special_tokens=False))

    return noun_phrases


def np_constrained_decode(model: PreTrainedModel,
                          input_ids: Tensor,
                          prepend_decoded_token_ids: List[int],
                          max_length: int,
                          min_length: Optional[int] = 0,
                          num_return_sequences: Optional[int] = 1,
                          num_beams: Optional[int] = 1,
                          no_repeat_ngram_size: Optional[int] = 0,
                          top_k: Optional[int] = 50,
                          top_p: Optional[float] = 1.0,
                          temperature: Optional[float] = 1.0,
                          repetition_penalty: Optional[float] = 1.0,
                          use_cache: Optional[bool] = True,
                          length_penalty: Optional[float] = 1.0,
                          bad_words_ids: Optional[Tensor] = None,
                          **model_specific_kwargs
                          ) -> torch.LongTensor:

    bos_token_id = model.config.bos_token_id
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id

    assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    batch_size = input_ids.shape[0]

    # create attention mask
    attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # current position and vocab size
    if hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    elif (
            model.config.is_encoder_decoder
            and hasattr(model.config, "decoder")
            and hasattr(model.config.decoder, "vocab_size")
    ):
        vocab_size = model.config.decoder.vocab_size

    effective_batch_size = batch_size * num_return_sequences
    effective_batch_mult = num_return_sequences

    bos_token_id = model.config.bos_token_id
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id

    decoder_start_token_id = bos_token_id

    encoder = model.get_encoder()

    encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)


    #
    # # Expand input ids to effective batch size if num_beams > 1 or num_return_sequences > 1
    # if num_return_sequences > 1 or num_beams > 1:
    #     input_ids_len = input_ids.shape[-1]
    #     input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
    #     attention_mask = attention_mask.unsqueeze(1).expand(
    #         batch_size, effective_batch_mult * num_beams, input_ids_len
    #     )
    #
    #     input_ids = input_ids.contiguous().view(
    #         effective_batch_size * num_beams, input_ids_len
    #     )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
    #     attention_mask = attention_mask.contiguous().view(
    #         effective_batch_size * num_beams, input_ids_len
    #     )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
    #
    # source_input_ids = input_ids
    # source_input_ids_list = input_ids.cpu().numpy().tolist()

    if model.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids_list = [decoder_start_token_id] + prepend_decoded_token_ids
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=next(model.parameters()).device)
        input_ids = input_ids.unsqueeze(0)

        cur_len = input_ids.shape[1]

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]

    assert (
        cur_len < max_length
    ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

    output = _generate_no_beam_search(
        model=model,
        input_ids=input_ids,
        cur_len=cur_len,
        max_length=max_length,
        min_length=min_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        batch_size=effective_batch_size,
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        use_cache=use_cache,
        model_specific_kwargs=model_specific_kwargs,
    )

    return output


def _generate_no_beam_search(
        model,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = (encoder_outputs, None) if encoder_outputs is not None else None

    while cur_len < max_length:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )

        outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]

        scores = postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
        if model._use_cache(outputs, use_cache):
            past = outputs[1]

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature

            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return input_ids


def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(
    scores,
    input_ids,
    no_repeat_ngram_size,
    bad_words_ids,
    cur_len,
    min_length,
    max_length,
    eos_token_id,
    repetition_penalty,
    batch_size,
    num_beams,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores, batch_size, num_beams, input_ids, repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def calc_banned_ngram_tokens(prev_input_ids: Tensor,
                             num_hypos: int,
                             no_repeat_ngram_size: int,
                             cur_len: int) -> List[Union[list, dict]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """ Filters a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def vocabulary_constraint_filtering(
    logits: Tensor,
    tokens_to_keep: Optional[torch.LongTensor],
    filter_value: float = -1e9,
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """
    Only keep the list of tokens specified in "tokens_to_keep" list.
    if len(tokens_to_keep) < min_tokens_to_keep, don't apply the filter

    :param logits:
    :param tokens_to_keep: A List of tokens to keep in generation
    :param filter_value:
    :param min_tokens_to_keep:
    :return:
    """
    assert logits.size()[0] == tokens_to_keep.size()[0], "tokens_to_keep should have the same first (batch size) dimension as logits"


    for i in range(logits.size()[0]):
        _mask = torch.ones_like(logits[i])
        _mask[tokens_to_keep[i]] = 0
        logits[i] += _mask * filter_value

    return logits


def _reorder_cache(past: Tuple,
                   beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)
