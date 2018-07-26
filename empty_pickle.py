#creates an empty pickle file (serialized binary object) to be read while creating indexes from corpus

import pickle
import pandas as pd

def create_empty_pickle():

	#dataframe to read news articles from csv file
	df = pd.read_csv('data_final.csv')
	#list of douments
	mydoclist = df['Index_Article'].tolist()

	#initialization empty data structures
	vocabulary = set()
	doc_tf_matrix =[]
	idf_vector = []
	tf_idf = []

	#creating pickle object and writing information into it
	pickle_out = open("data","wb")
	pickle.dump(mydoclist,pickle_out)
	pickle.dump(vocabulary,pickle_out)
	pickle.dump(doc_tf_matrix,pickle_out)
	pickle.dump(idf_vector,pickle_out)
	pickle.dump(tf_idf,pickle_out)
	#closing pickle object
	pickle_out.close()

