from bs4 import BeautifulSoup
import requests
import sys
import regex as re

"""
Parses names in the "What links to here"-page. The names are printed and can be piped into a file.
"""

def get_names():
    n_names = 0
    tag = sys.argv[1]
    url_base = "https://www.wikidata.org"
    url = "https://www.wikidata.org/w/index.php?title=Special:WhatLinksHere/" + tag + "&limit=5000"

    while True:
        try:
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")

            name_list = soup.find(id="mw-whatlinkshere-list")
            name_items = name_list.find_all("li")
        except:
            break

        for item in name_items:
            item_id = item.find(class_="wb-itemlink-id")
            item_label = item.find(class_="wb-itemlink-label")
            if item_id and item_label:
                try:
                    name = item_label.contents[0]
                except:
                    continue
                if not re.findall(":", name):
                    print(name)

        url_item = soup.find(text="next 5,000")
        if not url_item:
            break
        url = url_base + url_item.parent["href"]

get_names()
