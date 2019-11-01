import stanfordnlp
import networkx as nx
import spacy

INTERESTED_NOUN_POS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$']
INTERESTED_ADJ_POS = ['JJ', 'JJR', 'JJS']
INTERESTED_ADVERB_POS = ['RB', 'RBR', 'RBS']
CORRECT_ADVERB_DEPENDENCIES = ['neg', 'advmod']
CONJUNCTION = ['conj']
ROOT_NODE_INDEX = -1

G = nx.DiGraph()
nlp = stanfordnlp.Pipeline(
    lang="en", treebank="en_gum", processors="tokenize,mwt,pos,depparse"
)
spacy_nlp = spacy.load('en_core_web_sm')

def get_noun_adjective_pairs(review):
    doc = nlp(review)
    noun_chunks = spacy_nlp(review)
    noun_adjective_pairs = []
    for noun in noun_chunks:
        pass
    for sentence in doc.sentences:
        create_graph_from_stanford_words_obj(sentence.words)
        noun_adjective_pairs.extend(get_noun_adjective_pairs_from_graph(sentence.words))
        G.clear()
    return noun_adjective_pairs

def add_edge_from_first_node_to_second_node(first_node, second_node):
    if not G.has_node(first_node):
        G.add_node(first_node)
    if not G.has_node(second_node):
        G.add_node(second_node)
    G.add_edge(first_node, second_node)

def get_noun_adjective_pairs_from_graph(words_obj):
    """
    (noun, adjective) tuple will be returned
    """
    length_of_words_obj = len(list(words_obj))
    words = [words_obj[i].text for i in range(length_of_words_obj)]
    pos = [words_obj[i].xpos for i in range(length_of_words_obj)]
    noun_adjective_pairs = []
    noun_indexes =  G.successors(ROOT_NODE_INDEX)
    for noun_index in noun_indexes:
        nouns = []
        # Consecutive noun words are counted as one big noun word
        # The last index will be stored in the graph so iterate backwards
        noun_groupings = get_noun_groupings(words_obj, noun_index)
        indexes = list(G.successors(noun_index))
        adjective_indexes = [i for i in indexes if pos[i] in INTERESTED_ADJ_POS]
        # adjective_indexes = G.successors(noun_index)
        temp_noun_indexes = [i for i in indexes if pos[i] in INTERESTED_NOUN_POS]
        for adjective_index in adjective_indexes:
            adverb_indexes = list(G.successors(adjective_index))
            if len(adverb_indexes) == 0:
                noun_adjective_pairs.append((noun_groupings, words[adjective_index]))
                for temp_noun_index in temp_noun_indexes:
                    temp_noun_groupings = get_noun_groupings(words_obj, temp_noun_index)
                    noun_adjective_pairs.append((temp_noun_groupings, words[adjective_index]))
            else:
                adverbs = []
                adverb_indexes.sort()
                if adverb_indexes[-1] < adjective_index:
                    for i in range(adverb_indexes[0] - 1, -1, -1):
                        #Used to handle cases in sentences like 'Tan Ah Kao is very cool but not very calm'
                        #where not is attached to cool instead of calm
                        if pos[i] not in INTERESTED_ADVERB_POS:
                            break
                        adverb_indexes.insert(0, i)
                    for adverb_index in adverb_indexes:
                        adverbs.append(words[adverb_index])
                    adjective_groupings = ' '.join(adverbs)
                    adjective_groupings += " " + words[adjective_index]
                else:
                    adjective_groupings = words[adjective_index]
                noun_adjective_pairs.append((noun_groupings, adjective_groupings))
                for temp_noun_index in temp_noun_indexes:
                    temp_noun_groupings = get_noun_groupings(words_obj, temp_noun_index)
                    noun_adjective_pairs.append((temp_noun_groupings, adjective_groupings))
    return noun_adjective_pairs

def get_noun_groupings(words_obj, noun_index):
    length_of_words_obj = len(list(words_obj))
    words = [words_obj[i].text for i in range(length_of_words_obj)]
    pos = [words_obj[i].xpos for i in range(length_of_words_obj)]
    nouns = []
    for i in range(noun_index, -1, -1):
        if pos[i] not in INTERESTED_NOUN_POS:
            break
        nouns.append(words[i])
    nouns.reverse()
    if len(nouns) == 1:
        noun_groupings = nouns[0]
    else:
        noun_groupings = ' '.join(nouns)
    return noun_groupings

def create_graph_from_stanford_words_obj(words_obj):
    """
    first level after the root node will contain nouns.
    second level after the root node will contain adjectives.
    third level after the root node will contain adverbs.
    """
    length_of_words_obj = len(list(words_obj))
    predicted_heads = [words_obj[i].governor for i in range(length_of_words_obj)]
    predicted_dependencies = [words_obj[i].dependency_relation for i in range(length_of_words_obj)]
    pos = [words_obj[i].xpos for i in range(length_of_words_obj)]
    G.add_node(ROOT_NODE_INDEX)
    for i in range(length_of_words_obj):
        # predicted_heads always return 0 for root and return actual index + 1 for the rest of headers
        if predicted_heads[i] == 0:
            continue
        if pos[i] in INTERESTED_NOUN_POS:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_ADJ_POS:
                add_edge_from_first_node_to_second_node(i, head_index)
                add_edge_from_first_node_to_second_node(ROOT_NODE_INDEX, i)
            elif pos[head_index] in INTERESTED_NOUN_POS and predicted_dependencies[i] in CONJUNCTION:
                add_edge_from_first_node_to_second_node(head_index, i)
        if pos[i] in INTERESTED_ADJ_POS:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_NOUN_POS:
                add_edge_from_first_node_to_second_node(head_index, i)
                add_edge_from_first_node_to_second_node(ROOT_NODE_INDEX, head_index)
            # elif pos[head_index] in INTERESTED_ADJ_POS:

                # noun_index = predicted_heads[head_index] - 1
                # add_edge_from_first_node_to_second_node(G, ROOT_NODE_INDEX, noun_index)
                # add_edge_from_first_node_to_second_node(G, noun_index, head_index)
                # add_edge_from_first_node_to_second_node(G, noun_index, i)

                # noun_index = list(G.predecessors(head_index))[0]
                # add_edge_from_first_node_to_second_node(G, noun_index, i)
                # add_edge_from_first_node_to_second_node(G, ROOT_NODE_INDEX, noun_index)
        if pos[i] in INTERESTED_ADVERB_POS and predicted_dependencies[i] in CORRECT_ADVERB_DEPENDENCIES:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_ADJ_POS:
                add_edge_from_first_node_to_second_node(head_index, i)
    for i in range(length_of_words_obj):
        if pos[i] in INTERESTED_ADJ_POS:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_NOUN_POS:
                continue
            if G.has_node(i) and len(list(G.predecessors(i))) > 0 and pos[list(G.predecessors(i))[0]] in INTERESTED_NOUN_POS:
                continue
            while not (G.has_node(head_index) and len(list(G.predecessors(head_index))) > 0 and pos[list(G.predecessors(head_index))[0]] in INTERESTED_NOUN_POS):
                head_index = predicted_heads[head_index] - 1
                if head_index == -1:
                    break
            if head_index != -1:
                head_index = list(G.predecessors(head_index))[0]
                add_edge_from_first_node_to_second_node(head_index, i)



