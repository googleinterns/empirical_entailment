import json
import os
from typing import List, Dict
from torch.utils.data import Dataset


def make_jsonl_data(bbc_summary_data_dir: str,
                    split_file_path: str,
                    output_dir: str):
    with open(split_file_path) as fin:
        ds_split = json.load(fin)
        for split, filename in ds_split.items():
            output_path = os.path.join(output_dir, split + '.jsonl')
            print("Writing {} examples to {}...".format(len(filename), output_path))
            with open(output_path, 'w') as fout:
                for fn in filename:
                    summary_path = os.path.join(bbc_summary_data_dir, fn + ".summary")
                    if os.path.exists(summary_path):
                        with open(summary_path) as s_fin:
                            summary_lines = s_fin.readlines()
                            summary_json = parse_summary_file(summary_lines, fn)
                            summary_json_str = json.dumps(summary_json)

                            fout.write(summary_json_str)
                            fout.write('\n')

                    else:
                        print("Summary file not found: {}".format(fn))


def parse_summary_file(summary_file_lines: List[str],
                       summary_id: str):
    """
    parse a .summary file into json object
    :param summary_file_lines:
    :return: a json object containing information in the .summary file
    """

    # remove trailing newlines and spaces
    summary_file_lines = [l.strip() for l in summary_file_lines]
    summary_file_lines = [l for l in summary_file_lines if l]

    field_position = {}
    for idx, line in enumerate(summary_file_lines):
        if line.startswith("[SN]"):
            field = line.replace("[SN]", "").lower()
            field_position[field] = idx

    result_json = {"id": summary_id}
    if 'url' in field_position:
        result_json['url'] = summary_file_lines[field_position['url'] + 1]

    if 'title' in field_position:
        result_json['title'] = summary_file_lines[field_position['title'] + 1]

    if 'first-sentence' in field_position:
        result_json['first-sentence'] = summary_file_lines[field_position['first-sentence'] + 1]

    if 'restbody' in field_position:
        result_json['restbody'] = summary_file_lines[field_position['restbody'] + 1 : -1]

    return result_json


class XSumDataProcessor:

    @classmethod
    def get_train_examples(cls, data_dir):
        data_path = os.path.join(data_dir, 'train.jsonl')
        return cls._get_examples(data_path=data_path)

    @classmethod
    def get_test_examples(cls, data_dir):
        data_path = os.path.join(data_dir, 'test.jsonl')
        return cls._get_examples(data_path=data_path)

    @classmethod
    def get_dev_examples(cls, data_dir):
        data_path = os.path.join(data_dir, 'validation.jsonl')
        return cls._get_examples(data_path=data_path)

    @classmethod
    def _get_examples(cls, data_path):
        return XSumDataset(data_path)


class XSumDataset(Dataset):
    def __init__(self, data_jsonl_path):
        self.data = []
        with open(data_jsonl_path) as fin:
            for line in fin:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        print("Usage: python ... [bbc_summary_dir] [split_file] [output_dir]", file=sys.stderr)
        exit(1)

    make_jsonl_data(sys.argv[1], sys.argv[2], sys.argv[3])