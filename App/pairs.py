import json
import nltk
from nltk.corpus import stopwords
# import re
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


# download required libraries
nltk.download("punkt")
nltk.download("stopwords")
# stanfordnlp.download("en_gum")

# import data
review_dic = {}
data_file = "../Data/processed_data.json"

with open(data_file, "r") as json_file:
    review_dic = json.load(json_file)

nlp = stanfordnlp.Pipeline(
    lang="en", treebank="en_gum", processors="tokenize,mwt,pos,depparse"
)

txt = "The person from Missouri whom worked in the kitchen, in cool Calgary, is great."

# add 'space' before comma if it does not exist
# txt = txt.replace(",", " ,")
# print(txt)

doc = nlp(txt)

splitted_txt = txt.split

# remove 'ADP .. NN|NNP' from statement
phrases = []
for s in doc.sentences:
    for i in range(len(s.words)):
        phrase = ""
        word = s.words[i]
        if word.xpos == "IN":
            phrase += word.text + " "

            for j in range(int(word.index), int(s.words[-1].index) - 1):
                noun = s.words[j]
                if noun.xpos == "NNP" or noun.xpos == "NN":
                    phrase += noun.text
                    phrases.append(phrase)
                    break
                phrase += noun.text + " "

for phrase in phrases:
    txt = txt.replace(phrase, "")

phrases.append(txt)

for phrase in phrases:
    print(phrase)
