import pandas as pd

# printing the results of retrival in the desired format

# Article- 1 : Cost of doing business in India still high: JP Morgan
# Even as there is increased buoyancy in sentiment relating to doing business in India with industrialists such as Ratan Tata reposing their faith in Pr
# Read more : http://www.financialexpress.com/market/high-cost-of-doing-business-in-india-is-an-issue-emerging-markets-slowing-down-jp-morgan/864377/
# Document id : 54



def print_results(docs):
	df = pd.read_csv('check_data_final.csv')
	idx = 1
	for doc_id in docs:
		title = df.iloc[doc_id]['Title']
		article = df.iloc[doc_id]['Article']
		url = df.iloc[doc_id]['URL']
		print ("Article-",idx, ":", title)
		print (article[0:150])
		print ("Read more :",url)
		print ("Document id :", doc_id+1)
		print()
		idx += 1