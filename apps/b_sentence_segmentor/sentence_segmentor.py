from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import spacy


def sentence_segmentor(bundled_dataset):
    # parser
    parser = spacy.load("en_core_web_sm")
    parser.add_pipe(parser.create_pipe("sentencizer"))

    sentences_count_all = []
    for dataset in bundled_dataset:
        sentences_count_per_review = []
        for sentences in dataset:
            doc = parser(sentences)
            sentences_count_per_review.append(len(list(doc.sents)))
        sentences_count_all.append(sentences_count_per_review)

    colors = ["#FF0000", "#0000FF", "#008000", "#FFA500", "#323232"]

    for i in range(len(sentences_count_all)):
        counts = dict(Counter(sentences_count_all[i]).most_common())
        # export to file
        with open("out/b_segmented_sentences_" + str(i + 1) + ".csv", "w") as f:
            f.write("rank,num_of_sentences,count\n")
            j = 0
            for key, value in counts.items():
                f.write(str(j) + ",")
                f.write(str(key) + ",")
                f.write(str(value) + "\n")
                j += 1

        counts = OrderedDict(sorted(counts.items()))
        #  plot
        fig, ax = plt.subplots(figsize=(25, 12.5))
        ax.set_title("Sentence Segmentation Distribution: Rating " + str(i + 1))
        dict_keys = list(counts.keys())
        xtick_range = np.arange(dict_keys[0], dict_keys[-1] + 1)
        ax.set_xlim([dict_keys[0] - 1, dict_keys[-1] + 2])
        ax.set_xticks(xtick_range)
        ax.bar(counts.keys(), counts.values(), color=colors[i])
        for j, value in enumerate(counts.keys()):
            ax.text(value, counts[value], str(counts[value]), ha="center", va="bottom")
        fig.savefig("out/b_segmented_sentences_" + str(i + 1) + ".png")
        plt.close()
