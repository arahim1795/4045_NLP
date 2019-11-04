import json
import re


def decontracted(phrase):
    # specific
    phrase = re.sub(r"[Ww]on\'t", "will not", phrase)
    phrase = re.sub(r"[Cc]an\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    # phrase = re.sub(r"\'s", " is", phrase)
    # phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def split_review(phrase):
    phrases = phrase.split("\n")
    phrases = list(filter(None, phrases))
    return phrases


def merge_phrases(phrases):
    phrases = " ".join(phrases)
    return re.sub(r"  ", " ", phrases)


def converter(filename):
    review_dic = {}
    data_file = "data/" + filename + ".json"

    with open(data_file, "r") as json_file:
        reviews = json_file.readlines()

    for i in range(len(reviews)):
        review = json.loads(reviews[i])
        review["text"] = merge_phrases(split_review(decontracted(review["text"])))
        review_dic[i] = review
        i += 1

    with open("data/processed_data.json", "w") as json_file:
        json.dump(review_dic, json_file)


def import_data(filename):
    reviews = {}
    file = "data/" + filename + ".json"

    with open(file, "r") as json_file:
        reviews = json.load(json_file)

    return reviews


def bundle_sentences(dataset):
    rate_1, rate_2 = [], []
    rate_3, rate_4 = [], []
    rate_5, rate_all = [], []

    for key, value in dataset.items():
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
