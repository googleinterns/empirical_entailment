import os
import torch
import numpy as np
import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import InputExample, glue_convert_examples_to_features


def produce_summary(model, tokenizer, source_text, cuda):
    input_ids = torch.tensor(tokenizer.encode(source_text, truncation=True, add_special_tokens=True)).unsqueeze(0)

    if cuda:
        input_ids = input_ids.to('cuda')

    generated = model.generate(input_ids)
    gen_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    gen_text = gen_text.strip()
    return gen_text


entailment_label_set = ['C', 'N', 'E']


def get_entailment_label(model, tokenizer, premise, hypo, cuda):
    entailment_input = [InputExample(text_a=premise, text_b=hypo, guid="")]
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

        entailment_logits = entailment_logits[0].detach().cpu().squeeze(0).numpy()

    pred_label = 'E' if entailment_logits[2] > 0 else 'N'
    return pred_label


def evaluate_entailment(data_dir, model_dir, cuda=True, limit=2000):
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

    num_examples = min(len(source_lines), limit)
    for idx in tqdm.trange(num_examples):
        sl = source_lines[idx]
        tl = target_lines[idx]
        pred_summary = produce_summary(model, tokenizer, sl, cuda)
        entailment_source_label = get_entailment_label(entailment_model, entailment_tokenizer, sl, pred_summary, cuda)
        entailment_target_label = get_entailment_label(entailment_model, entailment_tokenizer, tl, pred_summary, cuda)

        if entailment_source_label == 'E':
            entail_source += 1

        if entailment_target_label == 'E':
            entail_gold += 1

        total += 1

    print("Total :\t{}".format(total))
    print("Number of predictions that is entailed by source text:\t{}".format(entail_source))
    print("Number of predictions that is entailed by target summary:\t{}".format(entail_gold))


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python ... [xsum_huggingface_dir] [model_dir]", file=sys.stderr)
        exit(1)

    xsum_huggingface_dir = sys.argv[1]
    _model_dir = sys.argv[2]

    evaluate_entailment(xsum_huggingface_dir, _model_dir)
