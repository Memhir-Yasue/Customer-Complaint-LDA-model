import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
new_stop_words = (['not','good','food','place','go',
'great','get','time','do','service','be','come','order','make',
'really','well','back','also','would','try','have','love',
'take','nice','want','thing','wife','friends','well','first','many',
'lot','something','anything','much','someone','husband','sure','right',
'amazing,','beautiful','boyfriend','girlfriend','mgm','impressed','wonderful',
'las','vegas','star','way','small','guy',])
stop_words_2= stop_words.union(new_stop_words)
stop_words
import re
import pandas as pd
import numpy as np
from pprint import pprint
# topic modelr
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline
import spacy
import os

import seaborn as sns

def prepare_document_ngram(df,X_col):
    # Covert to list
    def clean_data(df):
        dirty_data = df[str(X_col)].values.tolist()
        data = []
        for text in dirty_data:
            if type(text) is not float:
                data.append(text)
        return data

    data = clean_data(df)

    # Remove new line characters
    data = [re.sub('\s+', ' ', sentence) for sentence in data]
    # Remove distracting double quotes
    data = [re.sub("\'", "", sentence) for sentence in data]
    # Tokenize words and Clean-up text
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    data_words = list(sent_to_words(data))

    # Creating Bigram and Trigram models
    bigram = gensim.models.Phrases(data_words, min_count= 3, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stopwords, make Bigram snd Lemmatize
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags):
        # https://soacy.io/api/annotation
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stopwords, make Bigram snd Lemmatize
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_2] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags):
        # https://soacy.io/api/annotation
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop words
    data_words_nostops =  remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)



    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser','ner'])

    # Do lemmatitization keeping only noun, adj, vb, ADV
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN','ADJ'])
    data_lemmatized = remove_stopwords(data_lemmatized)

    print(data_lemmatized[:1])

    # 11. Create the Dictionary and Corpus needed for Topic Modeling


    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create corpus
    texts = data_lemmatized

    # term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus[:1])
    # Human readable format of corpus (term-frequency)
    [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
    return corpus, id2word, data_lemmatized

def prepare_document_nogram(df,X_col):
    # Covert to list
    def clean_data(df):
        dirty_data = df[str(X_col)].values.tolist()
        data = []
        for text in dirty_data:
            if type(text) is not float:
                data.append(text)
        return data

    data = clean_data(df)
    # Remove new line characters
    data = [re.sub('\s+', ' ', sentence) for sentence in data]
    # Remove distracting double quotes
    data = [re.sub("\'", "", sentence) for sentence in data]
    # Tokenize words and Clean-up text
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    data_words = list(sent_to_words(data))


    # Remove Stopwords, make Bigram snd Lemmatize
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words_2] for doc in texts]

    def lemmatization(texts, allowed_postags):
        # https://soacy.io/api/annotation
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop words
    data_words_nostops =  remove_stopwords(data_words)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser','ner'])

    # Do lemmatitization keeping only noun, adj, vb, ADV
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN','ADJ'])
    # data_lemmatized = remove_stopwords(data_lemmatized)
    # data_lemmatized = remove_stopwords(data_lemmatized)
    print(data_lemmatized[:1])

    # 11. Create the Dictionary and Corpus needed for Topic Modeling


    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create corpus
    texts = data_lemmatized

    # term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus[:1])
    # Human readable format of corpus (term-frequency)
    [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
    return corpus, id2word, data_lemmatized

def freq_words(doc, terms = 20):
    all_words = ' '.join([text for text in doc])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})
    # selecting top 20 most frquent words
    d = words_df.nlargest(columns="count", n = terms)
    sns.set(font_scale=2)
    plt.figure(figsize=(30,10))
    ax = sns.barplot(data=d, x = "word", y= "count", label = 'small')
    plt.xticks(rotation=45)
    ax.set(ylabel = 'count')
    plt.show()
    return fdist

def train_model(num_topic,corpus,id2word,data_lemmatized,chunksize,update_every,passes):

    # 12. Building the Topic Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topic,
                                                random_state=0,
                                                update_every=update_every,
                                                chunksize=chunksize,
                                                passes=passes,
                                                per_word_topics=True)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    return lda_model,doc_lda

def train_model_multicore(num_topic,corpus,data_lemmatized,id2word,chunksize,passes):
    # 12. Building the Topic Model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            num_topics=num_topic,
                                            id2word=id2word,
                                            random_state=0,
                                            chunksize=chunksize,
                                            passes=passes,
                                            )


    pprint(lda_model.print_topics(num_topic))
    doc_lda = lda_model[corpus]

    return lda_model,doc_lda

def see_topics(lda_model,corpus,id2word):
    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    return vis

def curr_coherence_val(dictionary, corpus, data_lemmatized, num_topics,id2word):
    mallet_path = '/Users/yasue/mallet-2.0.8/bin/mallet' # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,
                                                 num_topics=num_topics,id2word=id2word
                                                )
    # SHow topics
    pprint(ldamallet.show_topics(formatted=False))

    # Compute Coherence score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet,texts=data_lemmatized,
    dictionary = id2word, coherence = 'c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\n Coherence Score: ', coherence_ldamallet)

def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    mallet_path = '/Users/yasue/mallet-2.0.8/bin/mallet' # update this path
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        score = coherencemodel.get_coherence()
        print("Num of topic: ",num_topics," score: ",score)
        coherence_values.append(score)

    return model_list, coherence_values

def coherence_plot(start,limit,step,coherence_values):
    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
