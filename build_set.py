from pyPodcastParser.Podcast import Podcast
import requests

with open('./feeds_unique.txt') as f:
    rss_feeds = f.readlines()

for feed in rss_feeds:
    response = requests.get(feed)
    podcast = Podcast(response.content)
    print(podcast.description)
