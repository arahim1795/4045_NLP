from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
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


def plot_bar(common_dict, chart_title, output_filename, bar_color="blue"):
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    ax.set_title(chart_title)

    ax.bar(common_dict.keys(), common_dict.values(), color=bar_color)
    fig.savefig("out/c_" + str(output_filename) + ".png")
    plt.close()


def tokenise_stem(review_dic):
    # - part 1: tokenise and stem
    # -- define stemmer
    stemmer = SnowballStemmer("english")

    # -- parameters
    sentences_list = []
    sentences_list_with_stemming = []
    word_count_with_stemming = []
    word_count_without_stemming = []
    words_with_stemming = []
    words_without_stemming = []

    # -- extract sentences
    for key in review_dic:
        review = review_dic.get(key, {}).get("text")
        sentences_list.append(review)

    for i in range(len(sentences_list)):
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

    for i in range(len(sentences_list_with_stemming)):
        tokens = nltk.word_tokenize(sentences_list_with_stemming[i])
        for token in tokens:
            words_with_stemming.append(token)
        word_count_with_stemming.append(len(tokens))

    word_count_with_stemming_counter = Counter(word_count_with_stemming)
    word_count_without_stemming_counter = Counter(word_count_without_stemming)

    # - part 2: plotting
    # -- before stem
    plot_bar(
        word_count_without_stemming_counter,
        "Distribution Without Stemming",
        "distribution_without_stem",
        "blue",
    )

    # -- plot after stem
    plot_bar(
        word_count_with_stemming_counter,
        "Distribution With Stemming",
        "distribution_with_stem",
        "orange",
    )

    # - part 3: most common words
    words_without_stemming_adjusted = remove_unwanted_phrase(words_without_stemming)
    words_with_stemming_adjusted = remove_unwanted_phrase(words_with_stemming)

    words_without_stemming_counter = Counter(words_without_stemming_adjusted)
    words_with_stemming_counter = Counter(words_with_stemming_adjusted)

    # -- export most common words to file
    most_common_data = []
    most_common_data.append(dict(words_without_stemming_counter.most_common()))
    most_common_data.append(dict(words_with_stemming_counter.most_common()))

    filenames = ["common_before_stemming", "common_after_stemming"]

    for i in range(len(most_common_data)):
        with open("out/c_" + filenames[i] + ".csv", "w") as f:
            j = 1
            f.write("rank,word,count\n")
            for key, value in most_common_data[i].items():
                f.write(str(j) + ",")
                f.write(key + ",")
                f.write(str(value) + "\n")
                j += 1
