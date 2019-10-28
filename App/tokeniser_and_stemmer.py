from collections import Counter
import json
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from tqdm import tqdm

nltk.download("punkt")
nltk.download("stopwords")

# import data
review_dic = {}
data_file = "../Data/processed_data.json"

with open(data_file, "r") as json_file:
    review_dic = json.load(json_file)


# tokenise and stem
stemmer = SnowballStemmer("english")

sentences_list = []
sentences_list_with_stemming = []
word_count_with_stemming = []
word_count_without_stemming = []
words_with_stemming = []
words_without_stemming = []

# extract sentences
for key in review_dic:
    review = review_dic.get(key, {}).get("text")
    sentences_list.append(review)

# tokenise and stem
for i in tqdm(range(len(sentences_list))):
    tokens = nltk.word_tokenize(sentences_list[i])
    word_count_without_stemming.append(len(tokens))
    new_sentence_as_list = []
    for token in tokens:
        words_without_stemming.append(token)  # list of tokens before stemming
        new_sentence_as_list.append(
            stemmer.stem(token)
        )  # list of tokens after stemming
    new_sentence = " ".join(new_sentence_as_list)
    sentences_list_with_stemming.append(new_sentence)  # sentence list

for i in tqdm(range(len(sentences_list_with_stemming))):
    tokens = nltk.word_tokenize(sentences_list_with_stemming[i])
    for token in tokens:
        words_with_stemming.append(token)
    word_count_with_stemming.append(len(tokens))

word_count_with_stemming_counter = Counter(word_count_with_stemming)
word_count_without_stemming_counter = Counter(word_count_without_stemming)


# plot graph: type 1
def plot_bar(common_dict, chart_title, output_filename, bar_color="blue"):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    bars = plt.bar(common_dict.keys(), common_dict.values())
    for bar in bars:
        bar.set_color(bar_color)
    plt.title(chart_title)
    plt.savefig("../Data/" + str(output_filename) + ".png")
    plt.close()


plot_bar(
    word_count_without_stemming_counter,
    "Distribution Without Stemming",
    "distribution_without_stem",
    "blue",
)
plot_bar(
    word_count_with_stemming_counter,
    "Distribution With Stemming",
    "distribution_with_stem",
    "orange",
)

# * Top 20 Words (Before and After Stemming)
# * - iterative method to remove stopwords, punctuations, and other phrases ('s)
# * - filter method removed, takes too long compared with iterative method


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


words_without_stemming_adjusted = remove_unwanted_phrase(words_without_stemming)
words_with_stemming_adjusted = remove_unwanted_phrase(words_with_stemming)

words_without_stemming_counter = Counter(words_without_stemming_adjusted)
words_with_stemming_counter = Counter(words_with_stemming_adjusted)

# * Plotting & Saving Results


def plot_bar_with_val(common_dict, chart_title, output_filename, bar_color="blue"):
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    ax.set_title(chart_title)
    # ax.set_ylabel("Accuracy")
    # ax.set_xlabel("Epoch")

    ax.bar(common_dict.keys(), common_dict.values(), color=bar_color)
    xlocs, xlabs = plt.xticks()
    # xlocs = [i for i in range(len(common_dict.keys()))]
    for i, v in enumerate(common_dict.values()):
        ax.text(xlocs[i], v, str(v), ha="center", va="bottom")
    fig.savefig("../Data/" + str(output_filename) + ".png")
    plt.close()


# * - before stemming
common_without_stem = dict(words_without_stemming_counter.most_common(20))
plot_bar_with_val(
    common_without_stem,
    "20 Most Common Words Before Stemming",
    "common_without_stem",
    "blue",
)

# * - after stemming
common_with_stem = dict(words_with_stemming_counter.most_common(20))
plot_bar_with_val(
    common_with_stem,
    "20 Most Common Words After Stemming",
    "common_with_stem",
    "orange",
)
