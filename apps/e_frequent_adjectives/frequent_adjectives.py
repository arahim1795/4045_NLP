from collections import Counter
import math
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import stanfordnlp
import string


def remove_unwanted_phrase(input_list, ignore_special_case=True):
    stop_words = set(stopwords.words("english"))
    punctuations = string.punctuation
    other = ["...", "''", "``"]
    output_list = []
    if ignore_special_case:
        other.append("'s")
    for i in range(len(input_list)):
        word = input_list[i].lower()
        if word in stop_words or word in punctuations or word in other:
            continue
        else:
            output_list.append(word)
    return output_list


def tokenise(sentences_bundle):
    token_all, token_processed = [], []

    for i in range(len(sentences_bundle)):
        bundle = sentences_bundle[i]
        temp_tokens = []
        for sentence in bundle:
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                temp_tokens.append(token)
        token_all.append(temp_tokens)

    for tokens in token_all:
        token_processed.append(remove_unwanted_phrase(tokens))

    for i in range(len(token_processed)):
        with open("data/e_tokenised_" + str(i + 1) + ".txt", "w") as f:
            for token in token_processed[i]:
                f.write(token + "\n")


def extract_adjective(bundles):
    adj_all = []

    nlp = stanfordnlp.Pipeline(
        lang="en", treebank="en_gum", processors="tokenize,mwt,lemma,pos"
    )

    for i in range(len(bundles)):
        bundle = bundles[i]
        temp_adj = []
        for token in bundle:
            doc = nlp(token)
            for s in doc.sentences:
                for word in s.words:
                    if word.upos == "ADJ":
                        temp_adj.append(word.text.lower())
        adj_all.append(temp_adj)

    for i in range(len(adj_all)):
        with open("data/e_adjective_" + str(i + 1) + ".txt", "w") as f:
            for word in adj_all[i]:
                f.write(word + "\n")

    return adj_all


def import_txt(filename):
    text_data = []

    with open("data/e_" + filename, "r") as f:
        text_data = f.read().split("\n")[:-1]

    return text_data


def calculate_indicativeness(
    per_rating_adj_list, all_adj_list, per_rating_word_count, all_word_count
):
    # parameters
    cross_entropy_list = []

    # set up counters
    per_rating_adj_counter = Counter(per_rating_adj_list)
    all_adj_counter = Counter(all_adj_list)

    # create set of adjective list
    per_rating_adj_set = set(per_rating_adj_list)

    for adj in per_rating_adj_set:
        # calc 1:
        adj_count = all_adj_counter[adj]
        adj_prob = adj_count / all_word_count

        # calc 2:
        per_rating_adj_count = per_rating_adj_counter[adj]
        per_rating_adj_prob = per_rating_adj_count / per_rating_word_count

        # final calc:
        cross_entropy = per_rating_adj_prob * math.log10(per_rating_adj_prob / adj_prob)

        cross_entropy_list.append([adj, cross_entropy])

    return cross_entropy_list


def export_frequency_data(data):
    for i in range(len(data)):
        with open("out/e_frequent_" + str(i + 1) + ".csv", "w") as f:
            j = 1
            f.write("rank,word,frequency\n")
            for entry in data[i]:
                f.write(str(j) + ",")
                f.write(entry[0] + ",")
                f.write(str(entry[1]) + "\n")
                j += 1


def export_indicative_data(data):
    for i in range(len(data)):
        with open("out/e_indicative_" + str(i + 1) + ".csv", "w") as f:
            j = 1
            f.write("rank,word,indicativeness\n")
            for entry in data[i]:
                f.write(str(j) + ",")
                f.write(entry[0] + ",")
                f.write(str(entry[1]) + "\n")
                j += 1


def frequent_adjectives(bundled_data):
    # process data
    tokenise(bundled_data)
    adjectives = extract_adjective(bundled_data)

    adj_file_list = [
        "adjective_1.txt",
        "adjective_2.txt",
        "adjective_3.txt",
        "adjective_4.txt",
        "adjective_5.txt",
    ]
    token_file_list = [
        "tokenised_1.txt",
        "tokenised_2.txt",
        "tokenised_3.txt",
        "tokenised_4.txt",
        "tokenised_5.txt",
    ]

    adjectives, tokens = [], []
    adjectives_frequency = []

    for file in adj_file_list:
        data = import_txt(file)
        adjectives.append(data)
        adjectives_frequency.append(
            Counter(data).most_common()
        )

    for file in token_file_list:
        tokens.append(import_txt(file))

    adj_merged, token_merged = [], []
    for i in range(len(adj_file_list)):
        adj_merged += adjectives[i]
        token_merged += tokens[i]

    indicativeness = []
    for i in range(len(adjectives)):
        indicativeness.append(
            calculate_indicativeness(
                adjectives[i], adj_merged, len(tokens[i]), len(token_merged)
            )
        )

    # sort indicativeness descendingly
    sorted_indicativeness = []
    for calc_set in indicativeness:
        sorted_indicativeness.append(sorted(calc_set, key=itemgetter(1), reverse=True))

    # export data
    export_frequency_data(adjectives_frequency)
    export_indicative_data(sorted_indicativeness)
