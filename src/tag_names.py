import nltk
from nltk.tag.stanford import StanfordNERTagger
import sys
import regex as re

st = StanfordNERTagger(
    "/home/karvla/projects/topic-guest-classifier/english.all.3class.distsim.crf.ser.gz",
    "/home/karvla/projects/topic-guest-classifier/stanford-ner.jar",
)
batch_size = 9000
skipped = 0
def print_tagged(lines):
    text = "".join(lines)
    text_tokenized = nltk.word_tokenize(text, "english", True)
    nltk.word_tokenize

    name_parts = []

    names = False
    for token, tag in st.tag(text_tokenized):
        if tag == "PERSON" and not re.findall("\P{L}", token):
            name_parts.append(token)
        elif not tag == "PERSON" and len(name_parts) > 1:
            name = " ".join(name_parts)
            pattern = r"([^>])("+name+")([^<])"
            replace = r"\1<>\2<\>\3"
            text = re.sub(pattern, replace, text)
            name_parts = []

    if len(name_parts) > 1:
            name = " ".join(name_parts)
            pattern = r"([^>])("+name+")([^<])"
            replace = r"\1<>\2<\>\3"
            text = re.sub(pattern, replace, text)

    print(text, end="")


# parsed_episodes = sys.stdin.read()
lines = []
while True:
    line = sys.stdin.readline()
    lines.append(line)
    if len(lines) == batch_size:
        print_tagged(lines)
        lines = []

print("Number of skipped lines: " + str(skipped))