import nltk
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import csv



stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
stop = set(stopwords.words('english'))

def remove(line):
	l = line.splitlines()
	l1 = []
	l2 = []
	l3 = []
	l4 = []
	for string in l:
		l1.append(re.sub("i.*?;"," ",string))
	for string in l1:
		l2.append(re.sub("j.*?\("," ",string))
	for string in l2:
		l3.append(re.sub("\{.*?}"," ",string))
	for string in l3:
		l4.append(re.sub("\).*?;"," ",string))
	l4 =  ' '.join(l4)
	l4 =' '.join(re.findall("[a-zA-Z0-9.'%?()!@$']+", l4))
	return l4

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

def write_csv_header():
	with open('data_final.csv', 'a') as csvfile:
		fieldnames = ['Title', 'URL', 'Article', 'Index_Article']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

def write_csv_row(title, url, read_article, index_article):
	with open('data_final.csv', 'a') as csvfile:
		fieldnames = ['Title', 'URL', 'Article', 'Index_Article']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'Title':title, 'URL':url, 'Article':read_article, 'Index_Article':index_article})
	


write_csv_header()

df = pd.read_csv('data.csv')
for index,row in df.iterrows():
	print(index)
	title = df.loc[index]['Title']
	url = df.loc[index]['URL']
	read_article = remove(df.loc[index]['Article'])
	index_article = cleanup(read_article)
	write_csv_row(title, url, read_article, index_article)