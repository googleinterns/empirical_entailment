from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def calcuate_rouge_score(summary: str,
                         reference: str) -> dict:
    """
    Get the Rouge-1 and Rouge-2 score for a (summary, reference) pair

    See: https://github.com/danieldeutsch/sacrerouge

    :param summary:
    :param reference:
    :return: Example below
    {'rouge-1': {'recall': 50.0, 'precision': 57.143, 'f1': 53.333},
    'rouge-2': {'recall': 14.285999999999998, 'precision': 16.667, 'f1': 15.384999999999998}}

    """
    _score = scorer.score(summary, reference)
    print(_score)
    return _score