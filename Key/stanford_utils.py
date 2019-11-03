from named_entity_recognizer import get_ne_in_sent_from
import stanfordnlp

stanfordnlp.download("en_ewt")
nlp = stanfordnlp.Pipeline(
    lang="en", treebank="en_ewt", processors="tokenize,mwt,pos,depparse"
)

INTERESTED_NOUN_POS = ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$", "WP", "WP$"]
INTERESTED_ADJ_POS = ["JJ", "JJR", "JJS"]
INTERESTED_ADVERB_POS = ["RB", "RBR", "RBS"]
CORRECT_ADVERB_DEPENDENCIES = ["neg", "advmod"]


def get_length_of_words_obj(words_obj):
    length_of_words_obj = len(list(words_obj))
    return length_of_words_obj


def get_text_of_words_obj(words_obj):
    length_of_words_obj = get_length_of_words_obj(words_obj)
    texts = [words_obj[i].text for i in range(length_of_words_obj)]
    return texts


def get_predicted_pos_of_heads(words_obj):
    predicted_heads = get_predicted_heads_from_words_obj(words_obj)
    pos_of_predicted_heads = [
        words_obj[predicted_head - 1].xpos for predicted_head in predicted_heads
    ]
    return pos_of_predicted_heads


def get_predicted_heads_from_words_obj(words_obj):
    length_of_words_obj = get_length_of_words_obj(words_obj)
    predicted_heads = [words_obj[i].governor for i in range(length_of_words_obj)]
    return predicted_heads


def get_predicted_dependencies_from_words_obj(words_obj):
    length_of_words_obj = get_length_of_words_obj(words_obj)
    predicted_dependencies = [
        words_obj[i].dependency_relation for i in range(length_of_words_obj)
    ]
    return predicted_dependencies


def get_predicted_pos_from_words_obj(words_obj):
    length_of_words_obj = get_length_of_words_obj(words_obj)
    predicted_pos = [words_obj[i].xpos for i in range(length_of_words_obj)]
    return predicted_pos


# get indexes of multiple nouns to be treated as a singlular noun phrase
def get_noun_pairs_index(words_obj):
    predicted_heads = get_predicted_heads_from_words_obj(words_obj)
    predicted_dependencies = get_predicted_dependencies_from_words_obj(words_obj)
    predicted_pos = get_predicted_pos_from_words_obj(words_obj)
    noun_pairs_index = {}
    for i in range(get_length_of_words_obj(words_obj)):
        if (
            predicted_pos[i] in INTERESTED_NOUN_POS
            and predicted_dependencies[i] == "conj"
        ):
            predicted_heads_index = predicted_heads[i] - 1
            if predicted_heads_index in noun_pairs_index:
                noun_pairs_list = noun_pairs_index[predicted_heads_index]
                noun_pairs_list.append(i)
            else:
                noun_pairs_index[predicted_heads_index] = [i]
    return noun_pairs_index


# get indexes of multiple adjectives to be treated as a singular adjective phrase
def get_adjective_pairs_index(words_obj):
    predicted_heads = get_predicted_heads_from_words_obj(words_obj)
    predicted_dependencies = get_predicted_dependencies_from_words_obj(words_obj)
    predicted_pos = get_predicted_pos_from_words_obj(words_obj)
    adj_pairs_index = {}
    for i in range(get_length_of_words_obj(words_obj)):
        if (
            predicted_pos[i] in INTERESTED_ADJ_POS
            and predicted_dependencies[i] == "conj"
        ):
            predicted_heads_index = predicted_heads[i] - 1
            if predicted_heads_index in adj_pairs_index:
                adj_pairs_list = adj_pairs_index[predicted_heads_index]
                adj_pairs_list.append(i)
            else:
                adj_pairs_index[predicted_heads_index] = [i]
    return adj_pairs_index

# # get indexes of adjective and respective adverbs to be treated as a singular adjective phrase
# # not working as intended, hence deprecated
# def get_adjective_adverb_pairs(words_obj):
#     predicted_heads = get_predicted_heads_from_words_obj(words_obj)
#     predicted_dependencies = get_predicted_dependencies_from_words_obj(words_obj)
#     predicted_pos = get_predicted_pos_from_words_obj(words_obj)
#     predicted_pos_of_heads = get_predicted_pos_of_heads(words_obj)
#     adjective_adverb_pairs = {}
#     for i in range(get_length_of_words_obj(words_obj)):
#         if predicted_pos[i] in INTERESTED_ADVERB_POS and  \
#             predicted_dependencies[i] in CORRECT_ADVERB_DEPENDENCIES and \
#                 predicted_pos_of_heads[i] in INTERESTED_ADJ_POS:
#             predicted_heads_index = predicted_heads[i] - 1
#             if i > 0:
#                 """
#                 To handle cases like 'not very good' where 'not' will be attached to 'Noun'
#                 """
#                 previous_index = i - 1
#                 if predicted_dependencies[previous_index] in CORRECT_ADVERB_DEPENDENCIES and \
#                     previous_index not in adjective_adverb_pairs:
#                     adjective_adverb_pairs[predicted_heads_index] = [previous_index]
#             if predicted_heads_index in adjective_adverb_pairs:
#                 adverb_list = adjective_adverb_pairs[predicted_heads_index]
#                 adverb_list.append(i)
#             else:
#                 adjective_adverb_pairs[predicted_heads_index] = [i]
#     return adjective_adverb_pairs


def get_possible_adjective_index_list(adjective_pairs, index_to_look_at):
    if index_to_look_at in adjective_pairs:
        result = [index_to_look_at]
        result.extend(adjective_pairs[index_to_look_at])
        return result
    else:
        return [index_to_look_at]


def get_possible_nouns_index_list(noun_pairs, index_to_look_at):
    if index_to_look_at in noun_pairs:
        result = [index_to_look_at]
        result.extend(noun_pairs[index_to_look_at])
        return result
    else:
        return [index_to_look_at]


def find_first_index_of_adjective_in_between_nouns(first_index, last_index, pos_tags):
    current_index = first_index
    for i in range(first_index, last_index + 1):
        if pos_tags[i] in INTERESTED_ADJ_POS:
            break
        current_index += 1
    return current_index


def find_last_index_of_adjective_in_between_nouns(first_index, last_index, pos_tags):
    current_index = last_index
    for i in range(last_index, first_index - 1, -1):
        if pos_tags[i] in INTERESTED_ADJ_POS:
            break
        current_index -= 1
    return current_index


def get_first_index_of_nouns_in_between_adjective(first_index, last_index, pos_tags):
    current_index = first_index
    for i in range(first_index, last_index + 1):
        if pos_tags[i] in INTERESTED_NOUN_POS:
            break
        current_index += 1
    return current_index


def get_natural_language_phrase_for_noun(noun_index, pos, texts):
    """
    Definitely need work to be done.
    Right now its matching every noun that follows the noun that 
    gets matched with an adjective
    """
    natural_language_possible_indexes = [noun_index]
    for i in range(noun_index + 1, len(list(pos))):
        if pos[i] in INTERESTED_NOUN_POS and pos[i] == pos[noun_index]:
            natural_language_possible_indexes.append(i)
        else:
            break
    natural_language_texts = [texts[i] for i in natural_language_possible_indexes]
    return " ".join(natural_language_texts)


def get_natural_language_phrase_for_adj(adj_index, predicted_pos, texts):
    """
    Should be working fine.
    Might need some improvements.
    IDK lol
    """
    adjective_adverb_pairs_list = []
    adjective_adverb_pairs_list.append(adj_index)
    for i in range(adj_index - 1, -1, -1):
        if predicted_pos[i] in INTERESTED_ADVERB_POS:
            adjective_adverb_pairs_list.append(i)
        else:
            break
    adjective_adverb_pairs_list.reverse()
    adjective_adverb_pairs_list_in_natural_language = [
        texts[j] for j in adjective_adverb_pairs_list
    ]
    adjective_in_natural_language = " ".join(
        adjective_adverb_pairs_list_in_natural_language
    )
    return adjective_in_natural_language


def get_noun_adjective_pairs(words_obj, noun_pairs, adjective_pairs):
    predicted_heads = get_predicted_heads_from_words_obj(words_obj)
    # predicted_dependencies = get_predicted_dependencies_from_words_obj(words_obj)
    predicted_pos = get_predicted_pos_from_words_obj(words_obj)
    predicted_pos_of_heads = get_predicted_pos_of_heads(words_obj)
    texts = get_text_of_words_obj(words_obj)
    noun_adjective_pairs = []
    for i in range(get_length_of_words_obj(words_obj)):
        if (
            predicted_pos[i] in INTERESTED_ADJ_POS
            and predicted_pos_of_heads[i] in INTERESTED_NOUN_POS
        ) or (
            predicted_pos[i] in INTERESTED_NOUN_POS
            and predicted_pos_of_heads[i] in INTERESTED_ADJ_POS
        ):
            predicted_heads_index = predicted_heads[i] - 1
            if predicted_pos[i] in INTERESTED_ADJ_POS:
                adjectives_list = get_possible_adjective_index_list(adjective_pairs, i)
                noun_list = get_possible_nouns_index_list(
                    noun_pairs, predicted_heads_index
                )
            else:
                adjectives_list = get_possible_adjective_index_list(
                    adjective_pairs, predicted_heads_index
                )
                noun_list = get_possible_nouns_index_list(noun_pairs, i)
            noun_list.sort()
            adjectives_list.sort()
            if adjectives_list[0] < noun_list[0]:
                """
                If statement above is 
                used to check if the noun and adjectives are arranged as such:
                'Cool John'
                """
                first_adjective_index = find_first_index_of_adjective_in_between_nouns(
                    noun_list[0], noun_list[-1], predicted_pos
                )
                """
                The above statement is used to break up two nouns that might be connected together:
                Such as 'Cool John and great kid'
                where John and kid are a conjunctive pair
                """
                noun_list = [
                    noun_index
                    for noun_index in noun_list
                    if noun_index < first_adjective_index
                ]
                first_noun_index = get_first_index_of_nouns_in_between_adjective(
                    adjectives_list[0], adjectives_list[-1], predicted_pos
                )
                """
                The above statement is used to break up two adjectives that might be connected together:
                Such as 'The food is great and wine is delicious.'
                where great and delicious are a conjunctive pair
                """
                adjectives_list = [
                    adjective_index
                    for adjective_index in adjectives_list
                    if adjective_index < first_noun_index
                ]
            else:
                """
                Else statement above is
                used to check if the noun and adjectives are arranged as such:
                'John is Cool'
                """
                last_adjective_index = find_last_index_of_adjective_in_between_nouns(
                    noun_list[0], noun_list[-1], predicted_pos
                )
                """
                The above statement is used to break up two nouns that might be connected together
                by checking if there is an adjective between the connected nouns
                and returning the index of the last adjective so that the nouns after
                the last adjective can be used to match with the adjective that we will be using
                for matching purposes
                """
                noun_list = [
                    noun_index
                    for noun_index in noun_list
                    if noun_index > last_adjective_index
                ]
                first_noun_index = get_first_index_of_nouns_in_between_adjective(
                    adjectives_list[0], adjectives_list[-1], predicted_pos
                )
                """
                The above statement is used to break up two adjectives that might be connected together:
                Such as 'The food is great and wine is delicious.'
                where great and delicious are a conjunctive pair
                """
                adjectives_list = [
                    adjective_index
                    for adjective_index in adjectives_list
                    if adjective_index < first_noun_index
                ]
            nouns_in_natural_language = [
                get_natural_language_phrase_for_noun(j, predicted_pos, texts)
                for j in noun_list
            ]
            adjectives_in_natural_lanuage = [
                get_natural_language_phrase_for_adj(j, predicted_pos, texts)
                for j in adjectives_list
            ]
            for noun in nouns_in_natural_language:
                print(noun)
                ner_label = get_ne_in_sent_from(noun)
                print("ner:" + str(len(ner_label)))
                for adjective in adjectives_in_natural_lanuage:
                    noun_adjective_pairs.append((noun.lower(), adjective.lower()))
    return noun_adjective_pairs


def get_noun_adjective_pairs_from_reviews(review):
    doc = nlp(review)
    result = []
    for sentence in doc.sentences:
        words_obj = sentence.words
        noun_pairs = get_noun_pairs_index(words_obj)
        # adjective_adverb_pairs = get_adjective_adverb_pairs(words_obj)
        adjective_pairs = get_adjective_pairs_index(words_obj)
        noun_adjective_pairs = get_noun_adjective_pairs(
            words_obj, noun_pairs, adjective_pairs
        )
        # text_of_words = get_text_of_words_obj(words_obj)
        # print(" ".join(text_of_words))
        result.extend(noun_adjective_pairs)
        # print(noun_adjective_pairs)
    return result
