from collections import Counter
import json
import string
import matplotlib.pyplot as plt
import nltk
import stanfordnlp
# from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
# from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import math
from operator import itemgetter

# * Parameters
# * - Read JSON
data_file = '/Users/tohyue-sheng/Documents/GitHub/CZ4045/4045_NLP/Data/processed_data_test.json'
review_dic = {}

# * - Tokenise and Stem
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
# stemmer = LancasterStemmer()

sentences_list = []
sentences_list_with_stemming = []
word_count_with_stemming = []
word_count_without_stemming = []
words_with_stemming = []
words_without_stemming = []

# * Read JSON
with open(data_file, 'r') as json_file:
    review_dic = json.load(json_file)

# * Extract Sentences
for key in review_dic:
    review = review_dic.get(key, {}).get('text')
    sentences_list.append(review) #list of sentences regardless of reviews

# * Tokenise and Stem
for i in tqdm(range(len(sentences_list))):
    tokens = nltk.word_tokenize(sentences_list[i])
    word_count_without_stemming.append(len(tokens))
    new_sentence_as_list = []
    for token in tokens:
        words_without_stemming.append(token) #List of all tokens without stemming in
        new_sentence_as_list.append(stemmer.stem(token)) #List of all tokens with stemmming
    new_sentence = " ".join(new_sentence_as_list)
    sentences_list_with_stemming.append(new_sentence) #sentence list

for i in tqdm(range(len(sentences_list_with_stemming))):
    tokens = nltk.word_tokenize(sentences_list_with_stemming[i])
    for token in tokens:
        words_with_stemming.append(token)
    word_count_with_stemming.append(len(tokens))

word_count_with_stemming_counter = Counter(word_count_with_stemming)
word_count_without_stemming_counter = Counter(word_count_without_stemming)

# * Plot Graph


def plot_bar(common_dict, chart_title, output_filename, bar_color='blue'):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    bars = plt.bar(common_dict.keys(), common_dict.values())
    for bar in bars:
        bar.set_color(bar_color)
    plt.title(chart_title)
    plt.savefig('/Users/tohyue-sheng/Documents/GitHub/CZ4045/4045_NLP/Data/' + str(output_filename) + '.png')
    plt.close()


plot_bar(
    word_count_without_stemming_counter,
    'Distribution Without Stemming',
    'distribution_without_stem',
    'blue'
    )
plot_bar(
    word_count_with_stemming_counter,
    'Distribution With Stemming',
    'distribution_with_stem',
    'orange'
    )

# * Top 20 Words (Before and After Stemming)
# * - iterative method to remove stopwords, punctuations, and other phrases ('s)
# * - filter method removed, takes too long compared with iterative method


def remove_unwanted_phrase(input_list, ignore_special_case=True):
    stop_words = set(stopwords.words('english'))
    punctuations = string.punctuation
    other = ['...', '\'\'', '``']
    output_list = []   
    if (ignore_special_case):
        other.append('\'s')
    for i in tqdm(range(len(input_list))):
        word = input_list[i].lower()
        if (word in stop_words or word in punctuations or word in other):
            continue
        else:
            output_list.append(word)
    return output_list


words_without_stemming_adjusted = remove_unwanted_phrase(words_without_stemming)
words_with_stemming_adjusted = remove_unwanted_phrase(words_with_stemming)

words_without_stemming_counter = Counter(words_without_stemming_adjusted)
words_with_stemming_counter = Counter(words_with_stemming_adjusted)

# * Plotting & Saving Results


def plot_bar_with_val(common_dict, chart_title, output_filename, bar_color='blue'):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    bars = plt.bar(common_dict.keys(), common_dict.values())
    for bar in bars:
        bar.set_color(bar_color)
        y = bar.get_height()
        bar_width = bar.get_width()
        plt.text(bar.get_x()+(bar_width/2), y + 100, y, ha='center')
    plt.title(chart_title)
    plt.savefig('/Users/tohyue-sheng/Documents/GitHub/CZ4045/4045_NLP/Data/' + str(output_filename) + '.png')
    plt.close()


# * - before stemming
common_without_stem = dict(words_without_stemming_counter.most_common(20))
plot_bar_with_val(
    common_without_stem,
    '20 Most Common Words Before Stemming',
    'common_without_stem',
    'blue'
    )

# * - after stemming
common_with_stem = dict(words_with_stemming_counter.most_common(20))
plot_bar_with_val(
    common_with_stem,
    '20 Most Common Words After Stemming',
    'common_with_stem',
    'orange'
    )





def counters_for_cross_entropy(rating):
    ######## tokenise the all the sentences within the specific ratings ########
    sentences_list_ratings = []
    words_without_stemming_ratings = []

    if rating == -1:
        for key in review_dic:
            review = review_dic.get(key, {}).get('text')
            sentences_list_ratings.append(review)
    else:
        for key in review_dic:
            if review_dic.get(key, {}).get('stars') == rating:
                review = review_dic.get(key, {}).get('text')
                sentences_list_ratings.append(review)
        print(sentences_list_ratings)

    for i in tqdm(range(len(sentences_list_ratings))):
        tokens = nltk.word_tokenize(sentences_list_ratings[i])
        for token in tokens:
            words_without_stemming_ratings.append(token)  # List of all tokens without stemming in

    print(words_without_stemming_ratings)
    words_without_stemming_adjusted_ratings = remove_unwanted_phrase(words_without_stemming_ratings)
    num_words_in_reviews_ratings = len(words_without_stemming_adjusted_ratings)
    print(words_without_stemming_adjusted_ratings)

    ########### End Tokenisation#########

    ############ Take the tokens and only grab the adjectives out with POS tagging into adj_list ###########
    adj_list = []
    nlp = stanfordnlp.Pipeline(processors="tokenize,mwt,lemma,pos")
    for token in tqdm(words_without_stemming_adjusted_ratings):
        doc = nlp(token)
        for sent in doc.sentences:
            for wrd in sent.words:
                if wrd.upos == "ADJ":
                    adj_list.append(token)

    ############## End POS tagging and adj_list ##############

    ######## Evaluate the counters ################
    num_adj_in_reviews_ratings = len(adj_list)
    unique_adj_in_reviews_ratings = list(set(adj_list))
    adj_list_counter_reviews_ratings = Counter(adj_list)

    return num_adj_in_reviews_ratings, unique_adj_in_reviews_ratings, adj_list_counter_reviews_ratings
    #return num_words_in_reviews_ratings, unique_adj_in_reviews_ratings, adj_list_counter_reviews_ratings

    ########end #######



def cross_entropy(num_adj_in_reviews_ratings, unique_adj_in_reviews_ratings, adj_list_counter_reviews_ratings,
                  num_adj_in_reviews_all, unique_adj_in_reviews_all, adj_list_counter_reviews_all):

    cross_entropy_list = []

    print(adj_list_counter_reviews_ratings)

    for adj in unique_adj_in_reviews_ratings:
        count_specific_adj = adj_list_counter_reviews_all[adj]
        count_all_adj = num_adj_in_reviews_all

        prob_adj = count_specific_adj / count_all_adj

        count_specific_adj_in_ratings = adj_list_counter_reviews_ratings[adj]
        count_all_adj_in_ratings = num_adj_in_reviews_ratings

        prob_adj_given_ratings = count_specific_adj_in_ratings / count_all_adj_in_ratings

        cross_entropy = prob_adj_given_ratings * math.log10(prob_adj_given_ratings/prob_adj)

        cross_entropy_list.append([adj,cross_entropy])

    return cross_entropy_list




print("Testing with rating 1")
num_adj_in_reviews_1, unique_adj_in_reviews_1, adj_list_counter_reviews_1 = counters_for_cross_entropy(1.0)
print(unique_adj_in_reviews_1)
# num_adj_in_reviews_2, unique_adj_in_reviews_2, adj_list_counter_reviews_2 = counters_for_cross_entropy("2.0")
# num_adj_in_reviews_3, unique_adj_in_reviews_3, adj_list_counter_reviews_3 = counters_for_cross_entropy("3.0")
# num_adj_in_reviews_4, unique_adj_in_reviews_4, adj_list_counter_reviews_4 = counters_for_cross_entropy("4.0")
num_adj_in_reviews_5, unique_adj_in_reviews_5, adj_list_counter_reviews_5 = counters_for_cross_entropy(5.0)
print("Testing with all ratings")
num_adj_in_reviews_all, unique_adj_in_reviews_all, adj_list_counter_reviews_all = counters_for_cross_entropy(-1)


cross_entropy_list_1 = cross_entropy(num_adj_in_reviews_1, unique_adj_in_reviews_1, adj_list_counter_reviews_1,
                                     num_adj_in_reviews_all, unique_adj_in_reviews_all, adj_list_counter_reviews_all)

cross_entropy_list_5 = cross_entropy(num_adj_in_reviews_5, unique_adj_in_reviews_5, adj_list_counter_reviews_5,
                                     num_adj_in_reviews_all, unique_adj_in_reviews_all, adj_list_counter_reviews_all)



print("Cross Entropy for Rating 1:")
cross_entropy_list_1 = sorted(cross_entropy_list_1, key=itemgetter(1))
print(cross_entropy_list_1)
cross_entropy_list_5 = sorted(cross_entropy_list_5, key=itemgetter(1))
print("Cross Entropy for Rating 5:")
print(cross_entropy_list_5)
