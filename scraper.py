import urllib
import requests
import re
from bs4 import BeautifulSoup

query = "London" + " wikipedia"
query = query.replace(' ', '+')
URL = f"https://google.com/search?q={query}"

#google returns different results for desktop and mobile queries. Specifying desktop here!
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"

headers = {"user-agent" : USER_AGENT}
resp = requests.get(URL, headers=headers)

if resp.status_code == 200:
    soup = BeautifulSoup(resp.content, "html.parser")

# all_results = []

for g in soup.find_all('div', class_='r'):
    anchors = g.find_all('a')
    if anchors:
        URL = anchors[0]['href']
        title = g.find('h3').text
        # item = {
        #     "title": title,
        #     "link": link
        # }
        # all_results.append(item)
        match = re.search('wikipedia', URL)
        if(match):
            break

page = requests.get(URL)
wiki = BeautifulSoup(page.text, 'html.parser')

for i in wiki.select('p'):
    print(i.getText())