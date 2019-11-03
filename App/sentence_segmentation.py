import json
import spacy
from collections import Counter
import matplotlib.pyplot as plt

data_file = "../Data/reviewSamples20.json"

ratings_vs_sentence_count_dict = {}

with open(data_file, "r") as json_file:
    for line in json_file:
        review = json.loads(line)
        rating = review["stars"]
        sentences = review["text"]
        # run 'python -m spacy download en'
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentences)
        len_of_sentence = len(list(doc.sents))
        if rating in ratings_vs_sentence_count_dict:
            sentence_count_list = ratings_vs_sentence_count_dict[rating]
            sentence_count_list.append(len_of_sentence)
        else:
            ratings_vs_sentence_count_dict[rating] = [len_of_sentence]

ratings_list = list(ratings_vs_sentence_count_dict.keys())
ratings_list.sort()
for ratings in ratings_list:
    title = "sentence length vs count for ratings = " + str(ratings)
    sentence_count_list = ratings_vs_sentence_count_dict[ratings]
    sentence_count_list.sort()
    sentence_count_counter = Counter(sentence_count_list)

    plt.title(title)
    plt.bar(sentence_count_counter.keys(), sentence_count_counter.values())
    plt.show()
