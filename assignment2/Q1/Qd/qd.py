import numpy as np
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import os
import re
import sys

sw = set(STOPWORDS)
ps = PorterStemmer()

# model
class MultinomialEventModel:
    
    def __init__(self, n):
        self.n = n
        self.cond_params = [[] for i in range(n)]
        self.cond_params_denom = np.zeros(n)
        self.params = np.zeros(n)
        
        self.word_map = {}
        self.index_map = [] # will this be used?
        self.next_word_idx = 0

        # for wordcloud generation
        self.cond_param_freqs = [{} for i in range(n)]

    def fit(self, dataset):
        for (e,c) in dataset:
            self.params[c] += 1
            self.cond_params_denom[c] += len(e)
            for w in e:
                if w not in self.word_map:
                    self.word_map[w] = self.next_word_idx
                    self.index_map.append(w)
                    self.next_word_idx += 1
                    for l in self.cond_params:
                        l.append(0)
                idx = self.word_map[w]
                self.cond_params[c][idx] += 1
        
        for i in range(self.n):
            for j in range(len(self.cond_params[i])):
                self.cond_param_freqs[i][self.index_map[j]] = self.cond_params[i][j]
                self.cond_params[i][j] = (self.cond_params[i][j]+1)/(self.cond_params_denom[i]+self.next_word_idx)
            self.params[i] = self.params[i]/len(dataset)
        
    def predict(self, reviews):
        preds = np.zeros(len(reviews))
        for (j,r) in enumerate(reviews):
            p_list = np.log(self.params)
            for w in r:
                if w in self.word_map:
                    for i in range(self.n):
                        p_list[i] += np.log(self.cond_params[i][self.word_map[w]])
            preds[j] = np.argmax(p_list)
        
        return preds

def preprocess(line):
    break_rgx = re.compile(r"<[a-z ]*/>")
    line = break_rgx.sub(" ", line)
    punctuation_rgx = re.compile(r"[^a-zA-Z ]")
    line = punctuation_rgx.sub("", line)
    words = re.split(r"\s+", line.lower())
    new_words = []
    for word in words:
        if word not in sw:
            new_words.append(ps.stem(word))
    return new_words

def load_test_data(path):
    (pos, neg) = ([], [])

    for f in os.listdir(f'{path}/pos'):
        file = open(f'{path}/pos/{f}')
        review = preprocess(file.readlines()[0])
        pos.append(review)
        
    for f in os.listdir(f'{path}/neg'):
        file = open(f'{path}/neg/{f}')
        review = preprocess(file.readlines()[0])
        neg.append(review)

    return (pos, neg)

def load_train_data(path):
    dataset = []
    # 1 - positive, 0 - negative
    for f in os.listdir(f'{path}/pos'):
        file = open(f'{path}/pos/{f}')
        review = preprocess(file.readlines()[0])
        dataset.append((review,1))
        
    for f in os.listdir(f'{path}/neg'):
        file = open(f'{path}/neg/{f}')
        review = preprocess(file.readlines()[0])
        dataset.append((review,0))

    return dataset

def ri(a,b):
    return np.random.randint(a,b)

def green_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return f"rgb({ri(50,65)},{ri(180,220)},{ri(100,110)})"

def red_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return f"rgb({ri(160,180)},{ri(50,60)},{ri(100,110)})"

def make_wordcloud(frequencies, cf, save_name):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=56839
    ).generate_from_frequencies(frequencies)
    wordcloud.recolor(color_func = cf)
    wordcloud.to_file(save_name)

if __name__ == "__main__":
    train_data_path = sys.argv[1]
    train_dataset = load_train_data(train_data_path)
    model = MultinomialEventModel(2)
    model.fit(train_dataset)

    test_data_path = sys.argv[2]
    (testdata_pos, testdata_neg) = load_test_data(test_data_path)
    predcnt_pos = np.count_nonzero(model.predict(testdata_pos) == 1)
    predcnt_neg = np.count_nonzero(model.predict(testdata_neg) == 0)
    
    acc = (predcnt_pos+predcnt_neg)/(len(testdata_pos)+len(testdata_neg))
    print(f"# positive reviews correctly predicted: {predcnt_pos}")
    print(f"# negative reviews correctly predicted: {predcnt_neg}")
    print(f"Accuracy: ({predcnt_pos}+{predcnt_neg})/({len(testdata_pos)}+{len(testdata_neg)}) = {acc:.3f}")

    make_wordcloud(model.cond_param_freqs[1], green_color_func, "wc_pos.png")
    make_wordcloud(model.cond_param_freqs[0], red_color_func, "wc_neg.png")
