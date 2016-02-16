import numpy as np

def init_babi(fname):
    print "==> Loading test from %s" % fname
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": ""} 
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            task["C"] += line
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            tasks.append(task.copy())

    return tasks


def get_babi_raw(id, test_id):
    babi_map = [
        "qa1_single-supporting-fact",
        "qa2_two-supporting-facts",
        "qa3_three-supporting-facts",
        "qa4_two-arg-relations",
        "qa5_three-arg-relations",
        "qa6_yes-no-questions",
        "qa7_counting",
        "qa8_lists-sets",
        "qa9_simple-negation",
        "qa10_indefinite-knowledge",
        "qa11_basic-coreference",
        "qa12_conjunction",
        "qa13_compound-coreference",
        "qa14_time-reasoning",
        "qa15_basic-deduction",
        "qa16_basic-induction",
        "qa17_positional-reasoning",
        "qa18_size-reasoning",
        "qa19_path-finding",
        "qa20_agents-motivations",
        "../allen/ck12linesnoq",#21
        "MCTest",#22
        "19changed",#23
        "all_shuffled", #24,
    ]
    if (test_id == -1):
        test_id = id 
    babi_name = babi_map[int(id) - 1]
    babi_test_name = babi_map[int(test_id) - 1]
    babi_train_raw = init_babi('data/en/%s_train.txt' % babi_name)
    babi_test_raw = init_babi('data/en/%s_test.txt' % babi_test_name)
    return babi_train_raw, babi_test_raw

            
def load_glove(dim):
    word2vec = {}
    print "==> loading glove"
    with open("data/glove/glove.6B." + str(dim) + "d.txt") as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "%s is missing" % word
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)