from bs4 import BeautifulSoup
import requests
import sys

"""
Parses human names on wikidata and store them in a trie.
"""
sys.setrecursionlimit(500000)


def get_names():
    names = trie.TrieNode('*')

    n_names = 0

    url_base = "https://www.wikidata.org"
    url = (
        "https://www.wikidata.org/w/index.php?title=Special:WhatLinksHere/Q5&limit=5000"
    )

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
                print(name)

        url_item = soup.find(text="next 5,000").parent
        if not url_item:
            break
        url = url_base + url_item["href"]
        n_names += 5000
        
        if n_names % 1000000  == 0:
            with open("./names_on_wikipedia_" + str(n_names/1000000) + ".pickle", "wb") as f:
                pickle.dump(names, f)
            names = trie.TrieNode('*')

    return names


get_names()
