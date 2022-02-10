from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from util import parse_sts
import argparse
import numpy as np


def symmetrical_nist(text_pair):
    """
    Calculates symmetrical similarity as NIST(a,b) + NIST(b,a).
    :param text_pair: iterable to two strings to compare
    :return: a float
    """
    t1,t2 = text_pair

    # input tokenized text
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())

    # try / except for each side because of ZeroDivision Error
    # 0.0 is lowest score - give that if ZeroDivision Error
    try:
        nist_1 = sentence_nist([t1_toks, ], t2_toks)
    except ZeroDivisionError:
        # print(f"\n\n\nno NIST, {i}")
        nist_1 = 0.0

    try:
        nist_2 = sentence_nist([t2_toks, ], t1_toks)
    except ZeroDivisionError:
        # print(f"\n\n\nno NIST, {i}")
        nist_2 = 0.0

    return nist_1 + nist_2

def main(sts_data):
    """Calculate NIST metric for pairs of strings
    Data is formatted as in the STS benchmark"""

    # TODO 1: define a function to read the data in util
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # take a sample of sentences so the code runs fast for faster debugging
    # when you're done debugging, you may want to run this on more!
    sample_text = texts[120:140]
    sample_labels = labels[120:140]
    # zip them together to make tuples of text associated with labels
    sample_data = zip(sample_labels, sample_text)

    scores = []
    for label,text_pair in sample_data:
        print(label)
        print(f"Sentences: {text_pair[0]}\t{text_pair[1]}")
        # TODO 2: Calculate NIST for each pair of sentences
        # Define the function symmetrical_nist

        nist_total = symmetrical_nist(text_pair)
        print(f"Label: {label}, NIST: {nist_total:0.02f}\n")
        scores.append(nist_total)

    # This assertion verifies that symmetrical_nist is symmetrical
    # if the assertion holds, execution continues. If it does not, the program crashes
    first_pair = texts[0]
    text_a, text_b = first_pair
    nist_ab = symmetrical_nist((text_a, text_b))
    nist_ba = symmetrical_nist((text_b, text_a))
    assert nist_ab == nist_ba, f"Symmetrical NIST is not symmetrical! Got {nist_ab} and {nist_ba}"

    # TODO 3: find and print the sentences from the sample with the highest and lowest scores
    min_score_index = np.argmin(scores)
    min_score = scores[min_score_index]
    print(f"Lowest score: {min_score}")
    print(sample_text[min_score_index])
    assert min_score == symmetrical_nist(sample_text[min_score_index])

    max_score_index = np.argmax(scores)
    max_score = scores[max_score_index]
    print(f"Highest score: {max_score}")
    print(sample_text[max_score_index])
    assert max_score == symmetrical_nist(sample_text[max_score_index])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="sts data")
    args = parser.parse_args()

    main(args.sts_data)
