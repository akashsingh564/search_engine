import requests
import csv
from bs4 import BeautifulSoup

def write_csv_header():
	with open('data.csv', 'a') as csvfile:
		fieldnames = ['Title', 'URL','Article']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

def write_csv_row(title, url, article):
	with open('data.csv', 'a') as csvfile:
		fieldnames = ['Title', 'URL','Article']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'Title':title, 'URL':url,'Article':article})

def get_article(url, title):
	try:
		page = requests.get(url)
		soup = BeautifulSoup(page.content, 'html.parser')

		data = soup.find("div",{"class":"main-story-content"})
		paras = data.find_all("p")

		article = ""

		for para in paras:
			article = article + para.text

		print(title)
		# print(article)
		# print('\n')

		write_csv_row (title, url, article)
					
	except:
		write_csv_row (title, url, article)
		pass

	return; 


def read_list(current_url):
	print (current_url)
	try:
		page = requests.get(current_url)
		soup = BeautifulSoup(page.content, 'html.parser')

		data = soup.find("div",{"class":"listing"}).find("ul")
		list_items = data.find_all("li")

		for item in list_items:
			hyperlink = item.find("a")
			title = item.find("h5")
			
			# date = item.find("em").get_text()
			# l = date.split("|")
			# time_stamp = l[1]
			# time_stamp = time_stamp.strip()

			# print(title.get_text())
			# print(hyperlink.get('href'))
			# print(time_stamp)

			get_article(hyperlink.get('href'), title.get_text())	

	except:
		print ("An Error Has Occured while reading list")
		pass

	return; 

write_csv_header()

for num in range(1,500):
	url = "http://www.financialexpress.com/market/page/"+str(num)+"/"
	read_list(url)
