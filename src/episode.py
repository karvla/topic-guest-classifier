class Episode():

    def __init__(self, title, description):
        self.title = title
        self.description = description

    def print(self):
        print(self.title)
        print(self.description)

    def window(self, win_size=3):
        padding = ['NIL' for n in range(win_size)]
        words = padding + self.description.split() + padding
        window = None
        for i, word in enumerate(words):
            if word == 'NAME':
                window = []
                window.extend(words[i-win_size:i])
                window.extend(words[i:i+win_size+1])
        return window


        
        
        
