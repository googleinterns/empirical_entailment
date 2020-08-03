import spacy

annotator = spacy.load("en_core_web_sm")


def generate_subsentences(sentence: str):
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


if __name__ == '__main__':
    test_str = "A 20-year-old Romanian man had a lucky escape after stowing away on a plane flying from Vienna to London."
    print(generate_subsentences(test_str))