import urllib
import requests
import re
import pandas as pd
import wikipediaapi as wp

from bs4 import BeautifulSoup

#a script which searches the web for a particular item, goes to the wikipedia page for that item and downloads the first paragraph from the wiki page.

def get_wiki_para(query):
    query += " wikipedia"
    query = query.replace(' ', '+')
    URL = f"https://google.com/search?q={query}"

    #google returns different results for desktop and mobile queries. Specifying desktop here!
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"

    headers = {"user-agent" : USER_AGENT}
    resp = requests.get(URL, headers=headers)

    if resp.status_code == 200:
        #Beautiful soup library gets the web page content in a 'nice' format.
        soup = BeautifulSoup(resp.content, "html.parser")

    for g in soup.find_all('div', class_='r'):
        anchors = g.find_all('a')
        if anchors:
            URL = anchors[0]['href']
            title = g.find('h3').text
            #in all the weblinks in the first page of the google search results, go to the wikipedia link for the item. If a company/item doesn't have a wiki page,
            # will have to have a backup in place which scrapes a description paragraph from some other source.
            match = re.search('wikipedia', URL)
            if(match):
                break

    page = requests.get(URL)
    wiki = BeautifulSoup(page.text, 'html.parser')

    #return the first paragraph of the wiki page
    for i in wiki.select('p'):
        return i.getText()


def print_categories(page):
    categories = page.categories
    for title in sorted(categories.keys()):
        print("%s: %s" % (title, categories[title]))

# a function to get the a list of major the major cities in the world. If we have a list of all the cities, countries and regions, we can filter out the 
# 'locations' category.

def get_cities_list():
    URL = 'https://en.wikipedia.org/wiki/List_of_largest_cities'
    res = requests.get(URL).text
    soup = BeautifulSoup(res,'lxml')
    
    city_table = soup.find('table',{'class':'sortable wikitable mw-datatable'})

    city_names = []
    
    for row in city_table.find_all('tr')[1:]:
        state_cell = row.find_all('a')[0]  
        states.append(state_cell.text)
    print(states)

