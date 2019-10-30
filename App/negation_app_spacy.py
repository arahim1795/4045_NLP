
import json
import spacy
from tqdm import tqdm
import stanfordnlp


# * Parameters
# * - Read JSON
data_file = '../Data/processed_data_test.json'
review_dic = {}
review_list = []


def export_data(neg_sentences_list):

    filename = "../Data/neg_sents_results.txt"
    with open(filename, "w") as f:
        f.write("Number of negation sentences: " + str(len(neg_sentences_list)))
        for sentence in tqdm(neg_sentences_list):
            f.write("\n" + str(sentence))


with open(data_file, 'r') as json_file:
    review_dic = json.load(json_file)

# * Extract Sentences
for key in review_dic:
    review = (str(review_dic.get(key, {}).get('text'))).capitalize()
    review_list.append(review)


sentence_list = []

nlp = stanfordnlp.Pipeline(processors = "tokenize")

for review in review_list:
    doc = nlp(review)
    for i in range(0, len(doc.sentences)):
        sentence_list.append(" ". join(["{}".format(word.text) for word in doc.sentences[i].words]))

nlp = spacy.load('en_core_web_sm')
neg_sentences_list = []

## Uncomment the below three lines to test the printing of sample lines###
# doc = nlp("There were no newspapers left in the shop by one o’clock.")
# for token in doc:
#     print(token, token.dep_)


for sentence in tqdm(sentence_list):
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "neg":
            neg_sentences_list.append(sentence.capitalize())
            break


export_data(neg_sentences_list)