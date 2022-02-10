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
    nist_1 = 0.0
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
        print(f"Sentences: {texts[0]}\t{texts[1]}")
        # TODO 2: Calculate NIST for each pair of sentences
        # Define the function symmetrical_nist

        nist_total = symmetrical_nist(text_pair)
        print(f"Label: {label}, NIST: {nist_total:0.02f}\n")
        scores.append(nist_total)

    # This assertion verifies that symmetrical_nist is symmetrical
    # if the assertion holds, execution continues. If it does not, the program crashes
    first_pair = texts[0]
    print(first_pair)
    text_a, text_b = first_pair
    nist_ab = symmetrical_nist((text_a, text_b))
    nist_ba = symmetrical_nist((text_b, text_a))
    assert nist_ab == nist_ba, f"Symmetrical NIST is not symmetrical! Got {nist_ab} and {nist_ba}"

    # TODO 3: find and print the sentences from the sample with the highest and lowest scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="sts data")
    args = parser.parse_args()

    main(args.sts_data)
