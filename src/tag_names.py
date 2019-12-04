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
def tag_names(text):
    text_tokenized = nltk.word_tokenize(text, "english", True)
    nltk.word_tokenize
    name_parts = []
    names = False
    try:
        token_tags = st.tag(text_tokenized)
    except:
        return 
    for token, tag in token_tags:
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
    return text

def print_tagged(lines):
    text = "".join(lines)
    text = tag_names(text)

    print(text, end="")


# parsed_episodes = sys.stdin.read()
if __name__ == "__main__":
    lines = []
    while True:
        line = sys.stdin.readline()
        if line == "":
            break
        lines.append(line)
        if len(lines) == batch_size:
            print_tagged(lines)
            lines = []

