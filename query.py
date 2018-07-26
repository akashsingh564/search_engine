import string
import math
import nltk
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import csv
import pickle
from queue import PriorityQueue
from print_result import print_results
import time


#creating reverse priority queue by pushing -x for x into the priority queue
#user defined put and get functions
class ReversePriorityQueue(PriorityQueue):

	def put(self, tup):
	    newtup = tup[0] * -1, tup[1]
	    PriorityQueue.put(self, newtup)

	def get(self):
	    tup = PriorityQueue.get(self)
	    newtup = tup[0] * -1, tup[1]
	    return newtup
#readind data from pickle object for consistency in data,  de-serializing the data
pickle_in = open("data","rb")
mydoclist = pickle.load(pickle_in)
vocabulary = pickle.load(pickle_in)
doc_tf_matrix = pickle.load(pickle_in)
idf_vector = pickle.load(pickle_in)
tf_idf = pickle.load(pickle_in)
pickle_in.close()


# print (vocabulary)
# print (np.matrix(doc_tf_matrix))
# print (np.matrix(idf_vector))
# print (np.matrix(tf_idf))

#stemmer and lemmatizer object, list of english stop words
stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
stop = set(stopwords.words('english'))

#cleaning up the query, same as done for the documents
def cleanup(line):
	line = ' '.join(re.findall("[a-zA-Z']+", line))
	tokens= []
	line = nltk.word_tokenize(line.lower())
	for t in line:
		if("\'" in t):
			continue
		if(t not in stop):
			lem = lemmatizer.lemmatize(t)
			ste = stemmer.stem(lem)
			tokens.append(ste)
	tokens =  ' '.join(tokens)
	tokens =' '.join(re.findall("[a-zA-Z']+", tokens))
	return tokens

#
##### the functions below uses the modules from add_db_article.py #####
##### please refer to the same for documentation #####
#


def add_query(query):
	query_freq_vector = [tf(word, query) for word in vocabulary]
	tf_vector = tf_vectorize (query_freq_vector)
	return tf_vector

def tf(term, query):
  return freq(term, query)

def freq(term, query):
  return query.split().count(term)

def tf_vectorize(vec):
	query_tf_vector =[]
	for freq in vec:
		if(freq == 0):
			query_tf_vector.append(0)
		else:
			query_tf_vector.append(1+np.log(freq))

	return query_tf_vector

def tf_idf_vector(vec, idf_vector):
	noraml_vec = []
	for i in range(0,len(vec)):
		noraml_vec.append(vec[i]*idf_vector[i])

	unit_normal_vec = []
	d = np.sum([a**2 for a in noraml_vec])

	for a in noraml_vec:
		unit_normal_vec.append(a/math.sqrt(d)) 

	return unit_normal_vec

#taking query as input form the user
query = input("Please enter the query : ")

#noting the start time before processing the query
start_time = time.time()

#pre-processing the query
clean_query = cleanup(query)

#creating tf vector for the query
tf_vector = add_query(clean_query)
unit_normal_vec = tf_idf_vector(tf_vector, idf_vector)	

#reverse priority queue
pq = ReversePriorityQueue()
doc_id = 0

#computing similarity score of the query with each document
for doc in tf_idf:
	score = 0
	for i in range(0,len(doc)):
		score += (doc[i] * unit_normal_vec[i])
	pq.put((score, doc_id))
	doc_id = doc_id + 1

docs = []

#fetching top 10 matched documents
for i in range(0,10):
    next_item = pq.get()
    docs.append (next_item[1])

end_time = time.time()

#calling function to print the results
print_results(docs)

#printing the query processing time
time_taken = end_time - start_time
print ("Time taken : ", time_taken)