


import codecs

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
%config InlineBackend.figure_format='retina'
!apt -qq -y install fonts-nanum

import matplotlib.font_manager as fm
fontpath = '/user/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic')
mpl.font_manager._rebuild()
"""

def read_data(filename):
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

train_data = read_data('ratings_train.txt')

from konlpy.tag import Okt
tagger = Okt()

def tokenize(doc):
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

train_dosc = [row[1] for row in train_data]

sentences = [tokenize(d) for d in train_dosc]
print(sentences)

from gensim.models import word2vec

model = word2vec.Word2Vec(sentences)

model.wv.similarity(tokenize(u'배우 여배우'))