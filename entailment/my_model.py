import torch
import os
import sys
import tqdm
from transformers import BartModel, BartForConditionalGeneration, BartTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from data import XSumDataProcessor

# TODO: temporary to suppress a known bug with transformer package
# TODO: See https://github.com/huggingface/transformers/issues/5505
import logging
logging.basicConfig(level=logging.ERROR)

def train(data_dir: str,
          cuda: bool,
          num_epoch: int,
          bs: int,
          lr: float,
          num_warmup_steps: int,
          max_length: int = 1024):

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    entailment_model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

    if cuda:
        model.cuda()
        entailment_model.cuda()

    train_data = XSumDataProcessor.get_train_examples(data_dir=data_dir)

    sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=bs)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_training_step = len(train_dataloader) * num_epoch
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_step)

    global_step = 0
    model.zero_grad()
    model.train()

    for epoch in range(num_epoch):
        print("Epoch #{}".format(epoch))
        example_iter = tqdm.tqdm(train_dataloader, desc="Training Step")
        for step, batch in enumerate(example_iter):
            batch_original_text = batch['restbody']
            batch_original_text = [" ".join(sentences) for sentences in batch_original_text]

            batch_input = bart_tokenizer(batch_original_text,
                                         truncation=True,
                                         padding=True,
                                         max_length=max_length,
                                         return_tensors='pt')

            model.generate()
            break
        break


def is_cuda_available():
    """
    Check the presense or value of CUDA_VISIBLE_DEVICES to tell whether to use CUDA or CPU
    :return: True if CUDA_VISIBLE_DEVICES is set; False otherwise
    """
    cuda_device = os.getenv('CUDA_VISIBLE_DEVICES', default=None)
    if cuda_device:
        return True
    else:
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="directory containing train.jsonl")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=3)
    parser.add_argument("--num_warmup-steps", type=int, default=0)

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["cuda"] = is_cuda_available()
    train(**args_dict)


