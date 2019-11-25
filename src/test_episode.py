from episode import Episode, get_unlabeled, get_labeled

def test_names():
    title = 'This is a title.'
    description1 = 'This episode is featuring <>Arvid Larsson<\\> and <>Rickard Svensson<\\>'
    description2 = 'This episode is featuring <>Arvid Larsson<\\>'
    description3 = 'This episode is featuring no one'
    ep1 = Episode(title, description1)
    ep2 = Episode(title, description2)
    ep3 = Episode(title, description3)
    assert ep1.names() == ['Arvid Larsson', 'Rickard Svensson']
    assert ep2.names() == ['Arvid Larsson']
    assert ep3.names() == []

def test_tokenize():
    title = 'This is a title.'
    description1 = 'This episode is featuring <>Arvid Larsson<\\> and <>Rickard Svensson<\\>'
    description2 = 'This episode is featuring <>Arvid Larsson<\\> and <>Rickard Svensson<\\>'
    tokenized1 = title + '\n' + 'This episode is featuring NAME and OTHERS'
    tokenized2 = title + '\n' + 'This episode is featuring OTHERS and NAME'
    ep1 = Episode(title, description1)
    ep2 = Episode(title, description2)
    assert ep1.tokenize() == [(tokenized1, 'Arvid Larsson'), (tokenized2, 'Rickard Svensson')]

def test_getlabeled():
    labeled_txt = """Title
Desc
G

Title
Desc
T
"""
    print(labeled_txt.splitlines())

    episodes = get_labeled(labeled_txt)
    assert len(episodes) == 2
    assert episodes[0].guest == True
    assert episodes[1].guest == False

def test_getunlabeled():
    labeled_txt = """Title
Desc

Title
Desc
"""
    print(labeled_txt.splitlines())

    episodes = get_unlabeled(labeled_txt)
    assert len(episodes) == 2

