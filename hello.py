import nltk
import json
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import spacy

data = []

#read from json file
for line in open('reviewSelected100.json', 'r'):
    data.append(json.loads(line))

#do sentence segmentation
example = data[500]['text']
sentence_segmentation = sent_tokenize(example)
#print(sentence_segmentation)

#do word tokenizing
word_stemmer = PorterStemmer()
words = word_tokenize(example)

'''#without stemming
for word in words:
    print(word)

#with stemming
for word in words:
    print(word_stemmer.stem(word))'''

#list of stop words
stop_words = set(stopwords.words('english'))
#print(stop_words)
remove_stop_words = [w for w in words if not w in stop_words]
remove_stop_words = []

for w in words:
    if w not in stop_words:
        remove_stop_words.append(w)

#before removing stopwords
#print(words)
#after removing stopwords
#print(remove_stop_words)

#POS tagging
pos_tagging = nltk.pos_tag(words)
#print(pos_tagging)

#most frequent adjectives, first part
adjectives = [token[0] for token in pos_tagging if token[1] in ['JJ','JJR','JJS']]
#print(adjectives)
frequency = nltk.FreqDist(adjectives)
top_ten = frequency.most_common(10)
#print(top_ten)

#second part to be done below

#noun-adjective pair summarizer, dunno if can use spacy or not together with nltk
#from stack overflow
nlp = spacy.load('en_core_web_sm')

doc = nlp(example)
print(doc)
noun_adj_pairs = []
for adj,noun in enumerate(doc):
    if noun.pos_ not in ('NOUN','PROPN'):
        continue
    for j in range(adj+1,len(doc)):
        if doc[j].pos_ == 'ADJ':
            noun_adj_pairs.append((noun,doc[j]))
            break
print(noun_adj_pairs)


   


