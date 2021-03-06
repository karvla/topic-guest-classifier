import nltk
from nltk.tag.stanford import StanfordNERTagger
import sys
import regex as re

"""
Read lines from stdin and tagges all personal names like this  <>Arvid Larsson</>. The lines are
then printed and can be fed into a new file. The names are identified using the Stanford NER Tagger.
"""

# The following files are needed.
st = StanfordNERTagger(
    "/home/karvla/projects/topic-guest-classifier/english.all.3class.distsim.crf.ser.gz",
    "/home/karvla/projects/topic-guest-classifier/stanford-ner.jar",
)

batch_size = 1#9000
skipped = 0
def tag_names(text):
    text_tokenized = nltk.word_tokenize(text, "english", True)
    nltk.word_tokenize
    name_parts = []
    names = False

    def tagged_text(text, name_parts):
        full_name = " ".join(name_parts)
        replace = r"\1<>\2<\>\3"
        pattern = r"([^>])("+full_name+")([^<])"
        text = re.sub(pattern, replace, text)
        return text, []

    try:
        token_tags = st.tag(text_tokenized)
    except:
        return 

    for token, tag in token_tags:
        if tag == "PERSON" and not re.findall("\P{L}", token):
            name_parts.append(token)
        elif not tag == "PERSON" and len(name_parts) >= 2:
            text, name_parts = tagged_text(text, name_parts)

    if len(name_parts) > 1:
        text = tagged_text(text, name_parts)
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

