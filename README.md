Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence

NIST (nltk.translate.nist_score): Compute precision for n-grams of size 1 to 4 while giving more weight to correct, rarer n-grams 

BLEU (nltk.translate.bleu_score): Compute precision for n-grams of size 1 to 4 and giving equal weights to each one

WER (https://martin-thoma.com/word-error-rate-calculation/) : Compute the minimum number of editing steps to transform tokens from a string to match another string

LCS (difflib.SequenceMatcher): Find the longest string that is a substring of two or more strings

Edit Dist (nltk.edit_distance): Compute the minimum number of editing steps to transform characters of a string to match another string

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train | Dev | Test 
------ | ----- | --- | ----
NIST | 0.496 | 0.593 | 0.475
BLEU | 0.371 | 0.433 | 0.353
WER | -0.353 | -0.452| -0.358
LCS | 0.362 | 0.468| 0.347
Edit Dist | 0.033 | -0.175| -0.039

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).

python sts_pearson.py  --sts_data "stsbenchmark/sts-train.csv"

python sts_pearson.py  --sts_data "stsbenchmark/sts-dev.csv"

python sts_pearson.py  --sts_data "stsbenchmark/sts-test.csv"


## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.