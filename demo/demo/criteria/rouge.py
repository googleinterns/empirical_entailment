from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def calcuate_rouge_score(summary: str,
                         reference: str) -> dict:
    """
    Get the Rouge-1 and Rouge-L score for a (summary, reference) pair

    See: https://pypi.org/project/rouge-score/

    :param summary:
    :param reference:
    :return: Example below
    {'rouge-1': Score(recall=50.0, precision=57.143, fmeasure=53.333),
    'rouge-L': Score(recall=50.0, precision=57.143, fmeasure=53.333)}

    """
    _score = scorer.score(summary, reference)
    print(_score)
    return _score