from .mknn_utils import distance_matrix, validity, find_majority
from .exceptions import DistanceException
import numpy as np
import pandas as pd

class MKNN(object):
    def __init__(self, k=3, distance='euclidean' , k1=1.5, b=0.75):
        """
        Parameter
        ----------
        k\t= jumlah tetangga terdekat\n
        distance = 'euclidean', 'manhattan', 'cosine', 'bm25'

        *Noted: untuk distance bm25 harus menggunakan data 
        berupa list of string (corpus) sebagai feature-nya.
        """

        self.distance_list = [
            'euclidean',
            'manhattan',
            'cosine', 
            'bm25'
        ]

        self.k = k

        # Default parameter untuk BM25
        self.k1 = k1
        self.b = b

        if distance not in self.distance_list:
            raise DistanceException('jarak {} tidak dikenal'.format(distance))
        
        self.distance_index = self.distance_list.index(distance)
        
        self.distance_method = self.distance_list[self.distance_index]

    def fit(self, X, y):
        """
        Parameter
        ---------
        X : X training -> data training tanpa label\n
        y : y Training -> label data training\n
        """
        
        self.X_train = X
        if isinstance(y, pd.Series):
            self.y = y.values.ravel()
        else:
            self.y = y

        self.distance = distance_matrix(X, X, self.k1, self.b, self.distance_method)
        
        self.validity = validity(self.distance, self.y, self.k)


    def predict(self, X_test):
        """
        Parameter
        ----------
        X_test : data test Pandas DataFrame\n

        Return
        ----------
        list - hasil prediksi
        """
        if isinstance(X_test, pd.Series):
            test = X_test.values
        else:
            test = X_test
            
        predicted_label = []
        distances = distance_matrix(X_test, self.X_train, self.k1, self.b, self.distance_method)
        #print(distances)

        for i in distances:
            weight = []
            
            for j in range(len(self.validity)):
                weight_j = self.validity[j] * (1 / (i[j] + 0.5))
                weight.append(weight_j)
            
            sorted_index = sorted(range(len(weight)), key=lambda k: weight[k], reverse = True)
            f_label = []
            y = self.y
            for i in range(self.k):
                f_label.append(y[sorted_index[i]])
            
            majority, count = find_majority(f_label)
            #print(f_label)
            predicted_label.append(majority)

        return predicted_label
    
