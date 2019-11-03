import json
import nltk
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string

data_file = "../Data/test.json"

sentences_list = []
sentences_list_with_stemming = []
word_count_with_stemming = []
word_count_without_stemming = []
words_with_stemming = []
words_without_stemming = []

ps = PorterStemmer()

with open(data_file, "r") as json_file:
    for line in json_file:
        review = json.loads(line)
        sentences = review["text"]
        sentences_list.append(sentences)

for sentences in sentences_list:
    tokens = nltk.word_tokenize(sentences)
    word_count_without_stemming.append(len(tokens))
    new_sentence_as_list = []
    for token in tokens:
        words_without_stemming.append(token)
        new_sentence_as_list.append(ps.stem(token))
    new_sentence = " ".join(new_sentence_as_list)
    sentences_list_with_stemming.append(new_sentence)

for sentences in sentences_list_with_stemming:
    tokens = nltk.word_tokenize(sentences)
    for token in tokens:
        words_with_stemming.append(token)
    word_count_with_stemming.append(len(tokens))

word_count_with_stemming_counter = Counter(word_count_with_stemming)
word_count_without_stemming_counter = Counter(word_count_without_stemming)

plt.subplot(2, 1, 1)
plt.bar(
    word_count_with_stemming_counter.keys(), word_count_with_stemming_counter.values()
)

plt.subplot(2, 1, 2)
plt.bar(
    word_count_without_stemming_counter.keys(),
    word_count_without_stemming_counter.values(),
)

plt.show()


def filterWords(word):
    stop_words = set(stopwords.words("english"))
    if word.lower() in stop_words or word in string.punctuation:
        return False
    return True


words_with_stemming_counter = Counter(filter(filterWords, words_with_stemming))
words_without_stemming_counter = Counter(filter(filterWords, words_without_stemming))

print(
    "20 Most Common Words Before Stemming: "
    + str(words_without_stemming_counter.most_common(20))
)
print(
    "20 Most Common Words After Stemming: "
    + str(words_with_stemming_counter.most_common(20))
)

