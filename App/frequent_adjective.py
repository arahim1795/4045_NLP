from collections import Counter
import json
import math
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
        with open("../Data/tokenised_" + str(i) + ".txt", "w") as f:
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
        with open("../Data/adjective_" + str(i + 1) + ".txt", "w") as f:
            for word in adj_all[i]:
                f.write(word + "\n")

    return adj_all


def import_txt(filename):
    text_data = []
    
    with open("../Data/" + filename, "r") as f:
        text_data = f.read().split("\n")[:-1]
    
    return text_data


def cross_entropy(
    num_words_in_reviews_ratings,
    unique_adj_in_reviews_ratings,
    adj_list_counter_reviews_ratings,
    num_words_in_reviews_all,
    unique_adj_in_reviews_all,
    adj_list_counter_reviews_all,
):

    cross_entropy_list = []

    for adj in unique_adj_in_reviews_ratings:
        count_specific_adj = adj_list_counter_reviews_all[adj]
        count_all_words = num_words_in_reviews_all

        prob_adj = count_specific_adj / count_all_words

        count_specific_adj_in_ratings = adj_list_counter_reviews_ratings[adj]
        count_all_words_in_ratings = num_words_in_reviews_ratings

        prob_adj_given_ratings = (
            count_specific_adj_in_ratings / count_all_words_in_ratings
        )

        cross_entropy = prob_adj_given_ratings * math.log10(
            prob_adj_given_ratings / prob_adj
        )

        cross_entropy_list.append([adj, cross_entropy])

    return cross_entropy_list


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

adj_file_list = ["adjective_1.txt", "adjective_2.txt", "adjective_3.txt", "adjective_4.txt", "adjective_5.txt"]
token_file_list = ["tokenised_1.txt", "tokenised_2.txt", "tokenised_3.txt", "tokenised_4.txt", "tokenised_5.txt"]

adjectives, tokens = [], []

for file in adj_file_list:
    adjectives.append(import_txt(file))

for file in token_file_list:
    tokens.append(import_txt(file))


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


# cross_entropy_list_1 = cross_entropy(
#     num_words_in_reviews_1,
#     unique_adj_in_reviews_1,
#     adj_list_counter_reviews_1,
#     num_words_in_reviews_all,
#     unique_adj_in_reviews_all,
#     adj_list_counter_reviews_all,
# )

# cross_entropy_list_2 = cross_entropy(
#     num_words_in_reviews_2,
#     unique_adj_in_reviews_2,
#     adj_list_counter_reviews_2,
#     num_words_in_reviews_all,
#     unique_adj_in_reviews_all,
#     adj_list_counter_reviews_all,
# )

# cross_entropy_list_3 = cross_entropy(
#     num_words_in_reviews_3,
#     unique_adj_in_reviews_3,
#     adj_list_counter_reviews_3,
#     num_words_in_reviews_all,
#     unique_adj_in_reviews_all,
#     adj_list_counter_reviews_all,
# )

# cross_entropy_list_4 = cross_entropy(
#     num_words_in_reviews_4,
#     unique_adj_in_reviews_4,
#     adj_list_counter_reviews_4,
#     num_words_in_reviews_all,
#     unique_adj_in_reviews_all,
#     adj_list_counter_reviews_all,
# )

# cross_entropy_list_5 = cross_entropy(
#     num_words_in_reviews_5,
#     unique_adj_in_reviews_5,
#     adj_list_counter_reviews_5,
#     num_words_in_reviews_all,
#     unique_adj_in_reviews_all,
#     adj_list_counter_reviews_all,
# )


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
