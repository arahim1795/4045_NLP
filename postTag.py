from nltk.tokenize import sent_tokenize, word_tokenize
import json
import random

#read from json file
data = []
for line in open('reviewSelected100.json', 'r', encoding="utf-8"):
        data.append(json.loads(line))
#randomly select 5 lines
randomLines = random.sample(data, 5)
string_zero = randomLines[0]["text"]
string_one = randomLines[1]["text"]
string_two = randomLines[2]["text"]
string_three = randomLines[3]["text"]
string_four = randomLines[4]["text"]

#tokenize
charLength = 0
wordLength = 0
string_tokenized_list = sent_tokenize(string_one)

for x in range(0, len(string_tokenized_list)):
    for y in range(0, len(string_tokenized_list[x])):
        charLength+=1

test = word_tokenize(string_one)
for x in range(0, len(test)):
   wordLength+=1
print(string_tokenized_list)
print(charLength)
print(wordLength)
