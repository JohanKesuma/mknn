import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from rank_bm25 import BM25Okapi

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def distance_matrix(a, b, distance = 'euclidean'):
    if distance == 'euclidean':
        return euclidean_distances(a, b)
    elif distance == 'manhattan':
        return manhattan_distances(a, b)
    elif distance == 'cosine':
        return cosine_distances(a, b)
    elif distance == 'bm25':
        return bm25_distances(a, b)
    else:
        return False;

def bm25_distances(queries, corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    tokenized_queries = [list(set(doc.split())) for doc in queries]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_simmilarity = np.zeros(shape=(len(tokenized_queries), \
        len(tokenized_corpus)))
    
    for idx, query in enumerate(tokenized_queries):
        bm25_simmilarity[idx] = bm25.get_scores(query)
    
    # normalisasi nilai bm25 simmilarity
    max_simmilarity = np.max(bm25_simmilarity)
    bm25_simmilarity_norm = (bm25_simmilarity - 0) / (max_simmilarity - 0)

    # konversi nilai simmilarity menjadi distance
    bm25_distance = 1.0 - bm25_simmilarity_norm

    return bm25_distance

def validity(distance, y, k):
    v = []
    current_index = 0
    for i in distance:
        sorted_index = sorted(range(len(i)), key=lambda k: i[k])
        fk = []
        for j in range(k):
            fk.append(y[sorted_index[j + 1]]) # tidak termasuk dirinya sendiri
        
        same_label = 0
        #print('{} -> {}'.format(y[current_index], fk))
        for label in fk:
            if check_label(y[current_index], label):
                same_label += 1
        val = 1 / k * same_label
        v.append(val)
        current_index += 1

    return v    


def check_label(label1, label2):
    if label1 == label2:
        return True

    return False

