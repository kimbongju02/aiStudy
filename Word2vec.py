



import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
from nltk.corpus import movie_reviews
sentences = [list(s) for s in movie_reviews.sents()]
from gensim.models.word2vec import Word2Vec

model = Word2Vec(sentences)

model.init_sims(replace=True) # 모델 초기화

word_similarity = model.wv.similarity('he', 'she')
most_similar = model.wv.most_similar('home')
most_similar_PN = model.wv.most_similar(positive=['she', 'fruit'], negative='apple', topn=1)


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)

word = list(model.wv.index_to_key) # model의 단어 list
