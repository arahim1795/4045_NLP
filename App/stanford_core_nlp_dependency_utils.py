from stanfordnlp.server import CoreNLPClient

INTERESTED_NOUN_POS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$']
INTERESTED_ADJ_POS = ['JJ', 'JJR', 'JJS']
INTERESTED_ADVERB_POS = ['RB', 'RBR', 'RBS']
CORRECT_ADVERB_DEPENDENCIES = ['neg', 'advmod']

def get_noun_pairs_index(predicted_heads_and_dependencies):
    noun_pairs_index = {}
    for current_index, predicted_info_of_head in predicted_heads_and_dependencies.items():
        predicted_heads_index, predicted_dep, predicted_head_pos = predicted_info_of_head[0]
        if predicted_head_pos in INTERESTED_NOUN_POS and predicted_dep == "conj":
            if predicted_heads_index in noun_pairs_index:
                noun_pairs_list = noun_pairs_index[predicted_heads_index]
                noun_pairs_list.append(current_index)
            else:
                noun_pairs_index[predicted_heads_index] = [current_index]  
    return noun_pairs_index

def get_adjective_pairs_index(predicted_heads_and_dependencies):
    adj_pairs_index = {}
    for current_index, predicted_info_of_head in predicted_heads_and_dependencies.items():
        predicted_heads_index, predicted_dep, predicted_head_pos = predicted_info_of_head[0]
        if predicted_head_pos in INTERESTED_ADJ_POS and predicted_dep == "conj":
            if predicted_heads_index in adj_pairs_index:
                adj_pairs_list = adj_pairs_index[predicted_heads_index]
                adj_pairs_list.append(current_index)
            else:
                adj_pairs_index[predicted_heads_index] = [current_index]         
    return adj_pairs_index

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
    for i in range(noun_index - 1, -1, -1):
        if pos[i] in INTERESTED_NOUN_POS and pos[i] == pos[noun_index]:
            natural_language_possible_indexes.append(i)
        else:
            break
    natural_language_possible_indexes.sort()
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
    adjective_adverb_pairs_list_in_natural_language = [texts[j] for j in adjective_adverb_pairs_list]
    adjective_in_natural_language = " ".join(adjective_adverb_pairs_list_in_natural_language)
    return adjective_in_natural_language

def get_noun_adjective_pairs(predicted_heads_and_dependencies, predicted_pos, texts, noun_pairs, adjective_pairs):
    noun_adjective_pairs = [] 
    for i in range(len(predicted_pos)):
        if i not in predicted_heads_and_dependencies:
            continue
        for j in range(len(predicted_heads_and_dependencies[i])):
            predicted_heads_index, predicted_dep, predicted_head_pos = predicted_heads_and_dependencies[i][j]
            if (predicted_pos[i] in INTERESTED_ADJ_POS and predicted_head_pos in INTERESTED_NOUN_POS) or \
            (predicted_pos[i] in INTERESTED_NOUN_POS and predicted_head_pos in INTERESTED_ADJ_POS):
                if predicted_pos[i] in INTERESTED_ADJ_POS:
                    adjectives_list = get_possible_adjective_index_list(adjective_pairs, i)
                    noun_list = get_possible_nouns_index_list(noun_pairs, predicted_heads_index)
                else:
                    adjectives_list = get_possible_adjective_index_list(adjective_pairs, predicted_heads_index)
                    noun_list = get_possible_nouns_index_list(noun_pairs, i)
                noun_list.sort()
                adjectives_list.sort()
                if adjectives_list[0] < noun_list[0]:
                    """
                    If statement above is 
                    used to check if the noun and adjectives are arranged as such:
                    'Cool John'
                    """
                    first_adjective_index = find_first_index_of_adjective_in_between_nouns(noun_list[0], noun_list[-1], predicted_pos)
                    """
                    The above statement is used to break up two nouns that might be connected together:
                    Such as 'Cool John and great kid'
                    where John and kid are a conjunctive pair
                    """
                    noun_list = [noun_index for noun_index in noun_list if noun_index < first_adjective_index]
                    first_noun_index = get_first_index_of_nouns_in_between_adjective(adjectives_list[0], adjectives_list[-1], predicted_pos)
                    """
                    The above statement is used to break up two adjectives that might be connected together:
                    Such as 'The food is great and wine is delicious.'
                    where great and delicious are a conjunctive pair
                    """
                    adjectives_list = [adjective_index for adjective_index in adjectives_list if adjective_index < first_noun_index]
                else:
                    """
                    Else statement above is
                    used to check if the noun and adjectives are arranged as such:
                    'John is Cool'
                    """
                    last_adjective_index = find_last_index_of_adjective_in_between_nouns(noun_list[0], noun_list[-1], predicted_pos)
                    """
                    The above statement is used to break up two nouns that might be connected together
                    by checking if there is an adjective between the connected nouns
                    and returning the index of the last adjective so that the nouns after
                    the last adjective can be used to match with the adjective that we will be using
                    for matching purposes
                    """
                    noun_list = [noun_index for noun_index in noun_list if noun_index > last_adjective_index]
                    first_noun_index = get_first_index_of_nouns_in_between_adjective(adjectives_list[0], adjectives_list[-1], predicted_pos)
                    """
                    The above statement is used to break up two adjectives that might be connected together:
                    Such as 'The food is great and wine is delicious.'
                    where great and delicious are a conjunctive pair
                    """
                    adjectives_list = [adjective_index for adjective_index in adjectives_list if adjective_index < first_noun_index]
                nouns_in_natural_language = [get_natural_language_phrase_for_noun(j, predicted_pos, texts) for j in noun_list]
                adjectives_in_natural_lanuage = [get_natural_language_phrase_for_adj(j, predicted_pos, texts) for j in adjectives_list]
                for noun in nouns_in_natural_language:
                    for adjective in adjectives_in_natural_lanuage:
                        noun_adjective_pairs.append((noun, adjective))
    return noun_adjective_pairs

def get_noun_adjective_pairs_from_reviews_from_core_nlp(reviews):
    result = {}
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','depparse','lemma'], timeout=1000000000, memory='16G') as client:
        for review in reviews:
            ann = client.annotate(review['text'])
            business_id = review['business_id']
            for sentence in ann.sentence:
                dependency_parse = sentence.basicDependencies
                tokens = sentence.token
                predicted_heads_and_dependencies = {}
                predicted_pos = [] 
                predicted_lemm = []

                for i in range(len(tokens)):
                    predicted_pos.append(tokens[i].pos)
                    predicted_lemm.append(tokens[i].lemma)

                # predicted_dependencies = get_predicted_dependencies_from_words_obj(words_obj)
                for i in range(len(dependency_parse.edge)):
                    source = dependency_parse.edge[i].source
                    target = dependency_parse.edge[i].target
                    dep = dependency_parse.edge[i].dep
                    head_pos = predicted_pos[source - 1]
                    if target - 1 in predicted_heads_and_dependencies:
                        predicted_heads_and_dependencies_list = predicted_heads_and_dependencies[target - 1]
                        predicted_heads_and_dependencies_list.append((source - 1, dep, head_pos))
                    else:
                        predicted_heads_and_dependencies[target - 1] = [(source - 1, dep, head_pos)]
                noun_pairs = get_noun_pairs_index(predicted_heads_and_dependencies)
                adjective_pairs = get_adjective_pairs_index(predicted_heads_and_dependencies)
                noun_adjective_pairs = get_noun_adjective_pairs(predicted_heads_and_dependencies, predicted_pos, predicted_lemm, noun_pairs, adjective_pairs)
                if business_id in result:
                    actual_noun_adjective_pairs = result[business_id]
                    actual_noun_adjective_pairs.extend(noun_adjective_pairs)
                else:
                    result[business_id] = noun_adjective_pairs
    return result


