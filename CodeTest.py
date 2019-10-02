import json
from textblob import TextBlob,Word


file = open("reviewSelected5.json", 'r', encoding='utf-8')
datalist = []
i=1
dic = {}
for line in file.readlines():
    dic[i] = json.loads(line)  ##json.loads(line) gives the the string of each line in the json file. {reviewID:... date:...}
    i=i+1

for key in dic.keys():
    text = (dic[key]['text'])
    print("Original Text: "+ text + "\n")

    # Sentence Segmentation
    blob = TextBlob(text)
    for sentence in blob.sentences:
        print(sentence)
        # Tokenisation and Stemming, POS tagging
        for word, pos in sentence.tags:
            print(word + "/" + pos)
        print("\n")
        #Stemming/Lemenisation
        for word in sentence.words:
            print(Word(word).lemmatize())
        print("\n")
    print("\n")

    # ### For future references ###
    # for word,pos in blob.sentences[0].tags:
    #     print(word + "/" + pos)
    # print("\n")