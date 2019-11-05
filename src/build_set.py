from episode import Episode
import regex as re

with open('./all_episodes.txt') as f:
    all_episodes = f.read().splitlines()

print(len(all_episodes))
def episodes_with_name(name):
    episodes = []
    for title, description  in zip(all_episodes[0::2], all_episodes[1::2]):
        name_in_title = re.findall(name, title)
        name_in_description = re.findall(name, description)
        if name_in_description:
            title = re.sub(name, 'NAME', title)
            description = re.sub(name, 'NAME', description)
            ep = Episode(title, description)
            episodes.append(ep)
    return episodes

episodes = episodes_with_name("Sam Harris")
for ep in episodes:
    print(" ".join(ep.window()))

