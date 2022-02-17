# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """
    texts = []
    labels = []
    with open(data_file, 'r') as file:
        for line in file:
            fields = line.strip().split("\t")
            texts.append((fields[5].lower(), fields[6].lower()))
            labels.append(float(fields[4]))
    return texts, labels