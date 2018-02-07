import numpy as np
from sklearn.model_selection import train_test_split
import csv

import tensorflow as tf


def myWriteCsv(filename, data, classe) :
    nbEx = data.shape[0]
    nbCols = data.shape[1]

    f=open(filename,'w')
    chaine = str(nbEx)+","+str(nbCols)+"\n"
    f.write(chaine);

    for i in range(len(classe)):
        row = data[i]
        chaine = ""
        for elt in row:
            chaine+=str(elt)
            chaine+=','
        chaine+=str(int(classe[i]))
        chaine+='\n'
        f.write(chaine);
    f.close()
    

IRIS_ALL = "DATA/allExamples.csv"

myset = tf.contrib.learn.datasets.base.load_csv_with_header(
  filename=IRIS_ALL,
  target_dtype=np.int,
  features_dtype=np.float32)
  

xtrain, xtest, ytrain, ytest = train_test_split(myset.data, myset.target, test_size=0.20, random_state=42)



myWriteCsv("DATA/trainBase.csv", xtrain, ytrain)

myWriteCsv("DATA/testBase.csv", xtest, ytest)
