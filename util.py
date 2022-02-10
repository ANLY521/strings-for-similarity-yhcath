# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """
    texts = []
    labels = []

    with open(data_file, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1, t2))
    return texts, labels