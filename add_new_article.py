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


#
##### ***** the code below uses the modules from add_db_article.py ***** #####
##### please refer to the same for documentation #####
#

pickle_in = open("data","rb")
mydoclist = pickle.load(pickle_in)
vocabulary = pickle.load(pickle_in)
doc_tf_matrix = pickle.load(pickle_in)
idf_vector = pickle.load(pickle_in)
tf_idf = pickle.load(pickle_in)
pickle_in.close()

tf_idf = []

# print (vocabulary)
# print (np.matrix(doc_tf_matrix))
# print (np.matrix(idf_vector))
# print (np.matrix(tf_idf))

stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
stop = set(stopwords.words('english'))

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

def write_csv_row(title, url, read_article, index_article):
	with open('check_data_final.csv', 'a') as csvfile:
		fieldnames = ['Title', 'URL', 'Article', 'Index_Article']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'Title':title, 'URL':url, 'Article':read_article, 'Index_Article':index_article})

def build_lexicon(corpus):
    lexicon = set()
    for term in vocabulary:
    	lexicon.add(term)
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return sorted(lexicon)

def add_documnet(doc):
	doc_freq_vector = [tf(word, doc) for word in vocabulary]
	tf_vector = tf_vectorize (doc_freq_vector)
	return tf_vector

def tf(term, document):
  return freq(term, document)

def freq(term, document):
  return document.split().count(term)

def tf_vectorize(vec):
	doc_tf_vector =[]
	for freq in vec:
		if(freq == 0):
			doc_tf_vector.append(0)
		else:
			doc_tf_vector.append(1+np.log(freq))

	doc_tf_matrix.append(doc_tf_vector)
	return doc_tf_vector

def compute_idf():
	idf_vector = [idf(word, mydoclist) for word in vocabulary]
	return idf_vector

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / df)

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount += 1
    return doccount 

def compute_tf_idf(doc_tf_matrix, idf_vector):
	for vec in doc_tf_matrix:
		tf_idf_vector(vec, idf_vector)

def tf_idf_vector(vec, idf_vector):
	noraml_vec = []
	for i in range(0,len(vec)):
		noraml_vec.append(vec[i]*idf_vector[i])

	unit_normal_vec = []
	d = np.sum([a**2 for a in noraml_vec])

	for a in noraml_vec:
		unit_normal_vec.append(a/math.sqrt(d)) 

	tf_idf.append(unit_normal_vec)


title = input("Please enter the title of the article : ")
url = input("Please enter the URL of the article : ")
article = input("Please enter the article : ")

index_article = cleanup(article)
write_csv_row(title, url, article, index_article)

doclist = []
mydoclist.append(index_article)
doclist.append(index_article)
vocabulary = build_lexicon(doclist)

tf_vector = add_documnet(index_article)
idf_vector = compute_idf()
compute_tf_idf(doc_tf_matrix,idf_vector)
	

print (vocabulary)
print (np.matrix(doc_tf_matrix))
print (np.matrix(idf_vector))
print (np.matrix(tf_idf))

pickle_out = open("data","wb")
pickle.dump(mydoclist,pickle_out)
pickle.dump(vocabulary,pickle_out)
pickle.dump(doc_tf_matrix,pickle_out)
pickle.dump(idf_vector,pickle_out)
pickle.dump(tf_idf,pickle_out)
pickle_out.close()