from collections import Counter
import json
import math
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import stanfordnlp
import string
from tqdm import tqdm


def remove_unwanted_phrase(input_list, ignore_special_case=True):
    stop_words = set(stopwords.words("english"))
    punctuations = string.punctuation
    other = ["...", "''", "``"]
    output_list = []
    if ignore_special_case:
        other.append("'s")
    for i in tqdm(range(len(input_list))):
        word = input_list[i].lower()
        if word in stop_words or word in punctuations or word in other:
            continue
        else:
            output_list.append(word)
    return output_list


def bundle_sentences(data_dict):
    rate_1, rate_2 = [], []
    rate_3, rate_4 = [], []
    rate_5, rate_all = [], []

    for key, value in data_dict.items():
        eval_star = value.get("stars")
        sentences = value.get("text")
        if eval_star == 5:
            rate_5.append(sentences)
        elif eval_star == 4:
            rate_4.append(sentences)
        elif eval_star == 3:
            rate_3.append(sentences)
        elif eval_star == 2:
            rate_2.append(sentences)
        else:
            rate_1.append(sentences)

    rate_all.append(rate_1)
    rate_all.append(rate_2)
    rate_all.append(rate_3)
    rate_all.append(rate_4)
    rate_all.append(rate_5)

    return rate_all


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
        with open("../Out/tokenised_" + str(i + 1) + ".txt", "w") as f:
            for token in token_processed[i]:
                f.write(token + "\n")

    return token_processed


def extract_adjective(bundles):
    adj_all = []

    nlp = stanfordnlp.Pipeline(
        lang="en", treebank="en_gum", processors="tokenize,mwt,lemma,pos"
    )

    for i in tqdm(range(len(bundles))):
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
        with open("../Out/adjective_" + str(i + 1) + ".txt", "w") as f:
            for word in adj_all[i]:
                f.write(word + "\n")

    return adj_all


def import_txt(filename):
    text_data = []

    with open("../Out/" + filename, "r") as f:
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


def export_indicative_data(data, threshold):
    # csv export
    for i in range(len(data)):
        with open("../Out/indicative_" + str(i + 1) + ".csv", "w") as f:
            f.write("word,indicativeness\n")
            for entry in data[i]:
                f.write(entry[0] + "," + str(entry[1]) + "\n")

    # graph export
    # parameters
    color = ["#FF0000", "#0000FF", "#008000", "#FFA500", "#323232"]

    # - extract threshold amount
    words = []
    values = []
    for dataset in data:
        temp_word, temp_val = [], []
        for i in range(threshold):
            temp_word.append(dataset[i][0])
            temp_val.append(dataset[i][1])
        words.append(temp_word)
        values.append(temp_val)

    for i in range(len(data)):
        fig, ax = plt.subplots(figsize=(18.5, 10.5))
        ax.set_title("Indicativeness")

        ax.bar(words[i], values[i], color=color[i])
        xlocs, xlabs = plt.xticks()
        # for i, v in enumerate(common_dict.values()):
        #     ax.text(xlocs[i], v, str(v), ha="center", va="bottom")
        fig.savefig("../Out/indicative_" + str(i + 1) + ".png")
        plt.close()


# download required libraries
nltk.download("punkt")
nltk.download("stopwords")
# stanfordnlp.download("en_gum")

# import data
review_dic = {}
data_file = "../Data/processed_data.json"

with open(data_file, "r") as json_file:
    review_dic = json.load(json_file)

# # process data
# bundled_sentences = bundle_sentences(review_dic)
# tokenised_sentences = tokenise(bundled_sentences)
# adjectives = extract_adjective(bundled_sentences)

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

for file in adj_file_list:
    adjectives.append(import_txt(file))

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

# export indicative data
export_indicative_data(sorted_indicativeness, 10)

# def export_data(
#     list_1,
#     list_2,
#     list_3,
#     list_4,
#     list_5,
#     list_all,
#     cross_entropy_list_1,
#     cross_entropy_list_2,
#     cross_entropy_list_3,
#     cross_entropy_list_4,
#     cross_entropy_list_5,
# ):
#     adj_list = []
#     adj_list.append(list_1)
#     adj_list.append(list_2)
#     adj_list.append(list_3)
#     adj_list.append(list_4)
#     adj_list.append(list_5)
#     adj_list.append(list_all)

#     cross_entropy_list = []
#     cross_entropy_list.append(cross_entropy_list_1)
#     cross_entropy_list.append(cross_entropy_list_2)
#     cross_entropy_list.append(cross_entropy_list_3)
#     cross_entropy_list.append(cross_entropy_list_4)
#     cross_entropy_list.append(cross_entropy_list_5)
#     filename = "../Out/results.txt"
#     with open(filename, "w") as f:
#         for i in range(5):
#             f.write(
#                 "\nTop 10 most frequent adjectives for "
#                 + str(i + 1)
#                 + " star rating:\n"
#             )
#             f.write(str(adj_list[i].most_common(10)))
#             f.write(
#                 "\nTop 10 most indicative adjectives for "
#                 + str(i + 1)
#                 + " star ratings:\n"
#             )
#             f.write(str(cross_entropy_list[i][-10:][::-1]))


# # print("Top 10 adjectives for 1 star ratings:\n", adj_list_counter_reviews_1.most_common(10))
# # print("\nTop 10 adjectives for 2 stars ratings:\n", adj_list_counter_reviews_2.most_common(10))
# # print("\nTop 10 adjectives for 3 stars ratings:\n", adj_list_counter_reviews_3.most_common(10))
# # print("\nTop 10 adjectives for 4 stars ratings:\n", adj_list_counter_reviews_4.most_common(10))
# # print("\nTop 10 adjectives for 5 stars ratings:\n", adj_list_counter_reviews_5.most_common(10))


# cross_entropy_list_1 = sorted(cross_entropy_list_1, key=itemgetter(1))
# # print("\nTop 10 most indicative adjectives for 1 star ratings:\n", cross_entropy_list_1[-10:][::-1])

# cross_entropy_list_2 = sorted(cross_entropy_list_2, key=itemgetter(1))
# # print("\nTop 10 most indicative adjectives for 2 stars ratings:\n", cross_entropy_list_2[-10:][::-1])

# cross_entropy_list_3 = sorted(cross_entropy_list_3, key=itemgetter(1))
# # print("\nTop 10 most indicative adjectives for 3 stars ratings:\n", cross_entropy_list_3[-10:][::-1])

# cross_entropy_list_4 = sorted(cross_entropy_list_4, key=itemgetter(1))
# # print("\nTop 10 most indicative adjectives for 4 stars ratings:\n", cross_entropy_list_4[-10:][::-1])

# cross_entropy_list_5 = sorted(cross_entropy_list_5, key=itemgetter(1))
# # print("\nTop 10 most indicative adjectives for 5 stars ratings:\n", cross_entropy_list_5[-10:][::-1])
# export_data(
#     adj_list_counter_reviews_1,
#     adj_list_counter_reviews_2,
#     adj_list_counter_reviews_3,
#     adj_list_counter_reviews_4,
#     adj_list_counter_reviews_5,
#     adj_list_counter_reviews_all,
#     cross_entropy_list_1,
#     cross_entropy_list_2,
#     cross_entropy_list_3,
#     cross_entropy_list_4,
#     cross_entropy_list_5,
# )
