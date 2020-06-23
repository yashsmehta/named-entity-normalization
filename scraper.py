import urllib
import requests
import re
import pandas as pd
import wikipediaapi as wp

from bs4 import BeautifulSoup

def get_wiki_para(query):
    query += " wikipedia"
    query = query.replace(' ', '+')
    URL = f"https://google.com/search?q={query}"

    #google returns different results for desktop and mobile queries. Specifying desktop here!
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"

    headers = {"user-agent" : USER_AGENT}
    resp = requests.get(URL, headers=headers)

    if resp.status_code == 200:
        soup = BeautifulSoup(resp.content, "html.parser")

    for g in soup.find_all('div', class_='r'):
        anchors = g.find_all('a')
        if anchors:
            URL = anchors[0]['href']
            title = g.find('h3').text
            match = re.search('wikipedia', URL)
            if(match):
                break

    page = requests.get(URL)
    wiki = BeautifulSoup(page.text, 'html.parser')

    for i in wiki.select('p'):
        return i.getText()


def print_categories(page):
    categories = page.categories
    for title in sorted(categories.keys()):
        print("%s: %s" % (title, categories[title]))

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

