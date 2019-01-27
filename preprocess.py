import glob

import gensim
from nltk.corpus import stopwords


def cleantext(path, lang="english", savepath):
    corpus, target = [], []

    stopw = stopwords.words(lang)
    for file in glob(path):
        texts = open(file).read()
        i = 0
        for line in texts.splitlines():

            try:
                target.append(int(line.split("\t")[-1]))

                sc = gensim.utils.simple_preprocess(line)  # sentence candidate
                sc = [i for i in sc if i not in stopw]
                corpus.append(sc)
            except:
                print("encode error in: ", file, i)
            i += 1
    assert len(corpus) == len(target),\
     f"corpus: {len(corpus)} | target: {len(target)}"

    return corpus, target, stopw

def word2vec(corpus, size, sg, iterations, path):
    model = gensim.models.Word2Vec(
        corpus, size=size sg=sg,
        iter=iterations,  min_count=1
        )
        
    try:
        model.save_word2vec_format(path)
    except expression as identifier:
        print("error on save trained model")
    
    return model