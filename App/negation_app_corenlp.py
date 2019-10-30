import json
import spacy
from tqdm import tqdm
import stanfordnlp
from stanfordnlp.server import CoreNLPClient
import os

os.environ['CORENLP_HOME'] = "/Users/tohyue-sheng/Desktop/stanford-corenlp-full-2018-10-05" ### check to your path
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

# * Extract Reviews
for key in review_dic:
    review = (str(review_dic.get(key, {}).get('text'))).capitalize()
    review_list.append(review) #list of sentences regardless of reviews


## Extract Sentences
sentence_list = []
neg_sentences_list = []
nlp = stanfordnlp.Pipeline(processors = "tokenize")

for review in tqdm(review_list):
    doc = nlp(review)
    for i in range(0, len(doc.sentences)):
        sentence_list.append(" ". join(["{}".format(word.text) for word in doc.sentences[i].words]))


##Check Negation
print("Initiating Negation Check")

with CoreNLPClient(annotators=['tokenize','ssplit','pos','depparse'], timeout=60000, memory='16G') as client:

    for sent in tqdm(sentence_list):
        ann = client.annotate(sent)
        for sentence in ann.sentence:
            dependency_parse = sentence.basicDependencies
            for i in range(0, len(dependency_parse.edge)):
                dep = dependency_parse.edge[i].dep
                if dep == "neg":
                    neg_sentences_list.append(sent.capitalize())
                    break


#Export
export_data(neg_sentences_list)
