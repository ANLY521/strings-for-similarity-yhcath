from scipy.stats import pearsonr
import argparse
from util import parse_sts
from nltk import word_tokenize
from nltk import edit_distance
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
import difflib


def wer(r, h):
    """
    Found at https://martin-thoma.com/word-error-rate-calculation/
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time and space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int
    """
    # initialisation
    import numpy

    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    dataset = zip(labels, texts)
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]
    nist_scores = []
    bleu_scores = []
    word_error_rate_scores = []
    longest_common_substring_scores = []
    edit_distance_scores = []
    for label,text_pair in dataset:
        str1, str2 = text_pair
        tok1, tok2 = word_tokenize(str1.lower()), word_tokenize(str2.lower())
        # initialize scores
        nist_score = 0.0
        bleu_score = 0.0
        # get nist score
        try:
            nist_score += sentence_nist([tok1], tok2)
        except ZeroDivisionError:
            nist_score += 0.0
        try:
            nist_score += sentence_nist([tok2], tok1)
        except ZeroDivisionError:
            nist_score += 0.0
        # get bleu scores
        warnings.filterwarnings("ignore")
        try:
            bleu_score += sentence_bleu([tok1], tok2, smoothing_function=SmoothingFunction().method0)
        except ZeroDivisionError:
            bleu_score += 0.0
        try:
            bleu_score += sentence_bleu([tok2], tok1, smoothing_function=SmoothingFunction().method0)
        except ZeroDivisionError:
            bleu_score += 0.0
        # word error rate won't be symmetrical for all our samples, so we will treat it as symmetrical
        word_errors = wer(tok1, tok2)
        word_error_rate_score = word_errors/len(tok1) + word_errors/len(tok2)
        # get the longest substring
        sequence_matcher = difflib.SequenceMatcher(
            None, str1, str2)
        match = sequence_matcher.find_longest_match(
            0, len(str1), 0, len(str2))
        longest_common_substring_score = match.size
        # Finally, get the edit distance
        edit_distance_score = edit_distance(str1, str2)
        # Add all of the scores to their lists
        nist_scores.append(nist_score)
        bleu_scores.append(bleu_score)
        word_error_rate_scores.append(word_error_rate_score)
        longest_common_substring_scores.append(longest_common_substring_score)
        edit_distance_scores.append(edit_distance_score)

    scores = {
        "NIST": nist_scores,
        "BLEU": bleu_scores,
        "Word Error Rate": word_error_rate_scores,
        "Longest common substring": longest_common_substring_scores,
        "Edit Distance": edit_distance_scores
    }

    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:
        score = pearsonr(scores[metric_name], labels)
        print(f"{metric_name} correlation: {score[0]:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

