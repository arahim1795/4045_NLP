import json


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
