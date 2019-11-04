from collections import Counter
import stanfordnlp
from apps.f_noun_adjective_pair.stanford_utils import noun_adjective_pairer


nlp = stanfordnlp.Pipeline(processors="tokenize")


def summariser(num_business):
    # import data
    reviews_all = []
    for i in range(num_business):
        with open("data/f_business_" + str(i + 1) + ".txt", "r") as f:
            reviews_all.append(f.read().split("\n")[1:-1])

    counts_all = []
    for reviews in reviews_all:
        results = noun_adjective_pairer(reviews)
        counts_all.append(dict(Counter(results).most_common()))

    # export counts to file
    for i in range(len(counts_all)):
        with open("out/f_noun_adj_pair_" + str(i + 1) + ".csv", "w") as f:
            j = 1
            f.write("rank,noun,adj,count\n")
            for key, value in counts_all[i].items():
                f.write(str(j) + ",")
                for entry in key:
                    f.write(str(entry) + ",")
                f.write(str(value) + "\n")
                j += 1
