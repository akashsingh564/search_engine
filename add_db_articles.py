import string
import math
import numpy as np
import pickle
from empty_pickle import create_empty_pickle

#calling empty_picke.py to create empty pickle
create_empty_pickle()

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

#buidling vocabulary by putting each word from each document in a set 
def build_lexicon(corpus):
    lexicon = set()
    #copying terms from previous vocabulary
    for term in vocabulary:
    	lexicon.add(term)
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return sorted(lexicon)

#creating term_frequency matrix
#creates count matrix for the given corpus and return tf vector for the given document
def add_documnet(doc):
	doc_freq_vector = [tf(word, doc) for word in vocabulary]
	#converting count vector to tf vector
	tf_vector = tf_vectorize (doc_freq_vector)
	return tf_vector

#returns the frequency of given term in given document
def tf(term, document):
  return freq(term, document)

def freq(term, document):
  return document.split().count(term)

#creates term-frequency matrix for the given corpus and returns tf vector for the given document
#where tf = 1+log(word-frequency) if word-frequency != 0, 0 otherwise
def tf_vectorize(vec):
	doc_tf_vector =[]
	for freq in vec:
		if(freq == 0):
			doc_tf_vector.append(0)
		else:
			doc_tf_vector.append(1+np.log(freq))
	#appending tf vector for the given document to the tf matrix
	doc_tf_matrix.append(doc_tf_vector)
	return doc_tf_vector

#computes idf vector for all the terms in the corpus
def compute_idf():
	idf_vector = [idf(word, mydoclist) for word in vocabulary]
	return idf_vector

#computes idf for the given term
#idf = log(total documents in the corpus / total number of documents containg the given term)
def idf(word, doclist):
	print ("Constructing idf index for the term ", word)
	n_samples = len(doclist)
	df = numDocsContaining(word, doclist)
	return np.log(n_samples / df)

#returns number of documents containg the given word
def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount += 1
    return doccount 

#computes tf-idf matrix for the given corpus
def compute_tf_idf(doc_tf_matrix, idf_vector):
	i = 0
	for vec in doc_tf_matrix:
		print ("Constructing tf-idf for document number ", i)
		i += 1
		tf_idf_vector(vec, idf_vector)

#return if-idf vector for the given document
def tf_idf_vector(vec, idf_vector):
	noraml_vec = []
	for i in range(0,len(vec)):
		noraml_vec.append(vec[i]*idf_vector[i])

	#creating unit normal vector by dividing given vector by its length
	unit_normal_vec = []
	d = np.sum([a**2 for a in noraml_vec])

	for a in noraml_vec:
		unit_normal_vec.append(a/math.sqrt(d)) 

	tf_idf.append(unit_normal_vec)


tf_idf = []
#creating vocabulary for given corpus
vocabulary = build_lexicon(mydoclist)
print ("Done constructing Vocabulary")

#adding document one at a time
i = 0
for doc in mydoclist:
	print ("Adding doucument number ", i)
	tf_vector = add_documnet(doc)
	i += 1

#compute idf vector
idf_vector = compute_idf()
#compute tf-idf matrix
compute_tf_idf(doc_tf_matrix, idf_vector)

# print (vocabulary)
# print (np.matrix(doc_tf_matrix))
# print (np.matrix(idf_vector))
# print (np.matrix(tf_idf))
print ("Done constructing tf-idf matrix")

print ("Starting Pickle dump")
#dumping modified data structures on the pickle object, serializing the data
pickle_out = open("data","wb")
pickle.dump(mydoclist,pickle_out)
pickle.dump(vocabulary,pickle_out)
pickle.dump(doc_tf_matrix,pickle_out)
pickle.dump(idf_vector,pickle_out)
pickle.dump(tf_idf,pickle_out)
pickle_out.close()

print ("Completed Pickle dump")