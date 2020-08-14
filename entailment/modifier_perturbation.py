import random
from typing import List, Tuple, Dict
from allennlp.predictors.predictor import Predictor
import spacy
from spacy.tokens import Doc
from nltk import sent_tokenize

annotator = spacy.load("en_core_web_sm")

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")


def get_pos_annotation(tokenized_doc: List[str]) -> List[str]:
    """
    Gets the POS tags for a sequence of tokens
    :param tokenized_doc:
    :return:
    """
    _doc = Doc(annotator.vocab, words=tokenized_doc)
    for name, proc in annotator.pipeline:
        _doc = proc(_doc)

    return [tok.pos_ for tok in _doc]


def get_quantifiers_tokens(tokens: List[str],
                           pos_tags: List[str]) -> List[str]:
    """
    Gets all the quantifier or number tokens from a sequence of tokens
    :param tokens:
    :param pos_tags:
    :return:
    """
    assert len(tokens) == len(pos_tags), "length of tokens and pos_tags should be equal!"

    quants = []
    for idx, pos in enumerate(pos_tags):
        if pos == "NUM":
            quants.append(tokens[idx])

    return quants


def get_summary_modifier_span_offsets(summary_text: str) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    """
    Get the all modifier span offsets of a summary, predicted by a SRL model
    :param summary_text:
    :return: 1. a list of (modifier_label, span_start, span_end) 2. tokenized summary
    """
    srl_output = predictor.predict(sentence=summary_text)
    all_spans = []
    for verb in srl_output["verbs"]:
        span_list = []
        current_start = None
        in_span = False
        current_label = None
        for idx, lbl in enumerate(verb["tags"]):
            if not in_span: # currently not in a modifier span
                if lbl.startswith("B-ARGM"):
                    current_label = lbl[2:]
                    in_span = True
                    current_start = idx
            else:
                if lbl.startswith("I"):
                    continue
                else:
                    span_list.append((current_label, current_start, idx))
                    current_start = None
                    in_span = False

                    if lbl.startswith("B-ARGM"):
                        in_span = True
                        current_label = lbl[2:]
                        current_start = idx
        all_spans += span_list

    return all_spans, srl_output["words"]


def get_source_modifier_span_tokens(source_text: str) -> List[Tuple[str, List[str]]]:
    source_sents = sent_tokenize(source_text)
    all_spans = []
    for sent in source_sents:
        srl_output = predictor.predict(sentence=sent)
        for verb in srl_output['verbs']:
            span_list = []
            current_span = []
            in_span = False
            current_label = None
            for idx, lbl in enumerate(verb["tags"]):
                if not in_span: # currently not in a modifier span
                    if lbl.startswith("B-ARGM"):
                        current_label = lbl[2:]
                        in_span = True
                        current_span.append(srl_output["words"][idx])
                else:
                    if lbl.startswith("I"):
                        current_span.append(srl_output["words"][idx])
                    else:
                        span_list.append((current_label, current_span))
                        current_span = []
                        in_span = False

                        if lbl.startswith("B-ARGM"):
                            in_span = True
                            current_label = lbl[2:]
                            current_span.append(srl_output["words"][idx])
        all_spans += all_spans
    return all_spans


def _mod_list_to_dict(mods_list: List[Tuple[str, List[str]]]) -> Dict[str, List[List[str]]]:
    _dict = {}

    for lbl, toks in mods_list:
        if lbl not in _dict:
            _dict[lbl] = []

        _dict[lbl].append(toks)

    return _dict

def filter_source_mods(tag: str,
                       mod_toks: List[List[str]]) -> List[List[str]]:
    """
    Decides which modifiers should be excluded when making perturbations.
    :param tag: Modifier label
    :param mod_toks: tokens for hte modifier span
    :return:
    """
    if tag == "ARGM-ADV":
        return [toks for toks in mod_toks if len(toks) > 1]
    elif tag == "ARGM-MNR":
        return [toks for toks in mod_toks if len(toks) > 1]
    else:
        return mod_toks


def sample_mods(source_mods: list,
                sample_size: int = 3) -> list:
    """
    Sample k modifiers from a list of modifiers
    :param source_mods:
    :param sample_size:
    :return:
    """
    if len(source_mods) < sample_size:
        return source_mods
    else:
        return random.sample(source_mods, sample_size)


def generate_modifier_perturbations(source_text: str,
                                    summary_text: str) -> List[str]:
    """
    Generates a few perturbations on a summary of the source text, using modifiers seen in the source.
    :param source_text:
    :param summary_text:
    :return: A list of modified versions of the summary
    """
    all_perturbations = []

    summary_spans, summary_toks = get_summary_modifier_span_offsets(summary_text)

    source_modifiers = get_source_modifier_span_tokens(source_text)
    source_mod_dict = _mod_list_to_dict(source_modifiers)

    for lbl, span_start, span_end in summary_spans:
        if lbl not in source_mod_dict:
            continue

        source_mods_of_type = source_mod_dict[lbl]

        source_mods_of_type = filter_source_mods(lbl, source_mods_of_type)

        # filter out the same modifiers as the one in target summary
        source_mods_of_type = [mod for mod in source_mods_of_type if mod != summary_toks[span_start: span_end]]

        sampled_mods = sample_mods(source_mods_of_type, sample_size=3)
        for _m in sampled_mods:
            # print(_m, annotation["words"][span_start : span_end])
            recon_toks = summary_toks[: span_start] + _m + summary_toks[span_end:]
            recon_sentence = " ".join(recon_toks)

            all_perturbations.append(recon_sentence)

    # perturb quantifiers
    source_processed = annotator(source_text)
    source_toks = [tok.text for tok in source_processed]
    source_pos = [tok.pos_ for tok in source_processed]
    source_quants = get_quantifiers_tokens(tokens=source_toks,
                                           pos_tags=source_pos)
    orig_pos_tags = get_pos_annotation(summary_toks)
    for idx, tag in enumerate(orig_pos_tags):
        if tag == 'NUM':
            valid_perturb_quants = [q for q in source_quants if q != summary_toks[idx]]
            sampled_quants = valid_perturb_quants

            for _quants in sampled_quants:
                recon_toks = summary_toks[: idx] + [_quants] + summary_toks[idx + 1:]
                recon_sentence = " ".join(recon_toks)

                all_perturbations.append(recon_sentence)

    return all_perturbations
