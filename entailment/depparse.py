import spacy
from typing import List
annotator = spacy.load("en_core_web_sm")


def quantity_extraction(processed_doc) -> List:
    """
    Extracts Quantiites and number tokens from the a spacy processed document
    :param processed_doc:
    :return:
    """
    results = []
    for tok in processed_doc:
        if tok.pos_ == "NUM":
            _item = {
                "quantity": tok,
                "noun_phrase": None
            }
            # if the head token is noun and it's to the right of the number token, extract the whole noun phrase
            if (tok.head.pos_ in {"PROPN", "NOUN"}) and (tok.head.i > tok.i):
                _item["noun_phrase"] = [doc[idx] for idx in range(tok.i, tok.head.i + 1)]

            results.append(_item)
    return results


def generate_subsentences(sentence: str) -> List[str]:
    """
    Generates subsentences based on the modifiers on noun phrases.
    Example: If the original sentence include "The 18-year-old Paul....", then one of the subsentence gernerated by the
    function will be "Paul is 18-year-old"
    :param sentence:
    :return:
    """
    annotated_sent = annotator(sentence)
    merge_nps = annotator.create_pipe("merge_noun_chunks")

    generated_sub_sentences = []

    for chunk in annotated_sent.noun_chunks:
        span_left = chunk.root.left_edge
        span_right = chunk.root.right_edge
        noun_head = chunk.root

        # Generate subsentence for modifier to the left of the noun head
        if noun_head.i - span_left.i > 1 or (noun_head.i - span_left.i == 1 and span_left.pos_ != "DET"):
            sent = ["The"]
            sent += [tok.text for tok in annotated_sent[noun_head.i: span_right.i + 1]]
            if noun_head.pos_ == "NNS":
                sent.append("are")
            else:
                sent.append("is")

            if span_left.pos_ == "DET":
                left_offset = span_left.i + 1
            else:
                left_offset = span_left

            _modifier = [tok.text for tok in annotated_sent[left_offset: noun_head.i]]
            sent += _modifier
            generated_sub_sentences.append(" ".join(sent))

        # Generate subsentence for modifier to the right of the noun head
        if span_right.i - noun_head.i > 0:
            sent = ["The"]
            if span_left.pos_ == "DET":
                left_offset = span_left.i + 1
            else:
                left_offset = span_left

            sent += [tok.text for tok in annotated_sent[left_offset: noun_head.i + 1]]

            if noun_head.pos_ == "NNS":
                sent.append("are")
            else:
                sent.append("is")

            sent += [tok.text for tok in annotated_sent[noun_head.i + 1: span_right.i + 1]]

            generated_sub_sentences.append(" ".join(sent))

    return generated_sub_sentences