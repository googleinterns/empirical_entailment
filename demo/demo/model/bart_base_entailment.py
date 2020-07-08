import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_MODEL_DIR = f'models/bart_base_entailment/'

model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)

model = model.cuda()
model.eval()


def produce_summary(source_text):
    input_ids = torch.tensor(tokenizer.encode(source_text, add_special_tokens=True)).unsqueeze(0)
    generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
    gen_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    gen_text = gen_text.strip()
    return gen_text

if __name__ == '__main__':
    produce_summary("Brooks Brothers, the clothier that traces its roots to 1818, filed for bankruptcy. Harvard and M.I.T. sued the Trump administration over its plan to require foreign students to attend classes in person.")