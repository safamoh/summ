import os
import sys
import re
import codecs

import operator
from collections import defaultdict
from collections import OrderedDict

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.models import LdaModel
from gensim import models
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import strip_punctuation
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer # nltk.download('wordnet')

import pyLDAvis.gensim  #pip install pyLDAvis

from duc_reader import TEMP_DATA_PATH

ENC='utf-8'
sys.stdout = codecs.getwriter(ENC)(sys.stdout) #Allow special characters to be printed
sys.stderr = codecs.getwriter(ENC)(sys.stderr)
LOCAL_DIR = os.path.abspath(os.path.dirname(__file__))+"/"


#0v2# JC Sep  7, 2018  Integrate model viewer
#0v1# JC Aug 23, 2018  Setup for LDA topic modelling

##  REFERENCE:
#Options 1/
#{simple version}:  https://radimrehurek.com/gensim/models/ldamodel.html 
#Option 2/  
#{robust version}:  https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# Visualization (& pre-process options):
#https://www.kaggle.com/ykhorramz/lda-and-t-sne-interactive-visualization

TEMP_DATA_PATH=LOCAL_DIR+"../data/"
if not os.path.exists(TEMP_DATA_PATH):
    TEMP_DATA_PATH="/Users/safaalballaa/Desktop/resulted_files/"
if not os.path.exists(TEMP_DATA_PATH):
    print ("Error in configuration of temp directory: "+str(TEMP_DATA_PATH))

VERSION="DUC_2005"
NUM_TOPICS=40
MODEL_FILENAME = datapath(TEMP_DATA_PATH+"lda_"+str(NUM_TOPICS)+"_"+str(VERSION)+".model")
TOPIC_DICTIONARY_FILENAME=datapath(TEMP_DATA_PATH+"lda_dictionary_"+str(NUM_TOPICS)+"_"+str(VERSION)+".dict")
TOPIC_CORPUS_FILENAME=datapath(TEMP_DATA_PATH+"lda_corpus_"+str(NUM_TOPICS)+"_"+str(VERSION)+".mm")

def get_corpora(documents=[],common_dictionary='',verbose=True):
    from duc_reader import files2sentences #inner import
    
    if common_dictionary:
        flag_use_pretrained_dict=True
    else:
        flag_use_pretrained_dict=False

    #NOTES:
    #> if using for unseen documents -- use previous common dictionary

    #https://radimrehurek.com/gensim/tut1.html
    if not documents:
        documents,sentences,sentences_duc_topics=files2sentences(limit=-1) #watch loading 2x data into mem

    if verbose:
        print ("Loaded "+str(len(documents))+" documents.")
    print ("Extra pre-processing steps for Topic model...")
    
    # FILTER WORDS FOR TOPIC MODELLING
    #############################################33
    
    #1/   remove common words and tokenize
    #stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in STOPWORDS] for document in documents]

    #//  Remove punctuation (like commas at end of words)
    texts = [[strip_punctuation(token) for token in doc if not token.isdigit()] for doc in texts]
    
    #2/  Remove numbers, but not words that contain numbers.
    texts = [[token for token in doc if not token.isdigit()] for doc in texts]
    
    #3/  Remove words that are only one character.
    texts = [[token for token in doc if len(token) > 3] for doc in texts]
    
    #4/  Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    texts = [[lemmatizer.lemmatize(token) for token in doc] for doc in texts]
    
    #5/  Add n-grams
    #Since topics are very similar what would make distinguish them are phrases rather than single/individual words.
    
    if flag_use_pretrained_dict: #Then assume processing subset of new docs:
        bigram = Phrases(texts, min_count=1)
    else:
        bigram = Phrases(texts, min_count=10)

    trigram = Phrases(bigram[texts])
    for idx in range(len(texts)):
        for token in bigram[texts[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                texts[idx].append(token)
        for token in trigram[texts[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                texts[idx].append(token)
    
    # Create a corpus from a list of texts
    if not flag_use_pretrained_dict:
        common_dictionary = Dictionary(texts)  #common_texts)
        #7/  Filter out words that occur less than 10 documents, or more than 20% of the documents.
        if verbose: print('Number of unique words in initital documents:', len(common_dictionary))
        common_dictionary.filter_extremes(no_below=10, no_above=0.2)
    else:
        print ("[debug] using previously trained dictionary..")
        #must retrain model#  common_dictionary.add_documents(texts)
    
    common_corpus = [common_dictionary.doc2bow(text) for text in texts]
    
    if verbose:
        print('Number of unique tokens: %d' % len(common_dictionary))
        print('Number of documents: %d' % len(common_corpus))

    return documents,texts,common_corpus,common_dictionary

def train_model_basic():
    global MODEL_FILENAME,NUM_TOPICS,VERSION
    global TOPIC_DICTIONARY_FILENAME,TOPIC_CORPUS_FILENAME
    
    #common_texts : list of list of str
    print ("Loading corpora...")
    documents,texts,common_corpus,common_dictionary=get_corpora()

    # Train the model on the corpus.
    print ("Train lda model...")
    lda = LdaModel(common_corpus, num_topics=NUM_TOPICS)

    # Save all for visuals
    lda.save(MODEL_FILENAME)
    common_dictionary.save(TOPIC_DICTIONARY_FILENAME)
    corpora.MmCorpus.serialize(TOPIC_CORPUS_FILENAME, common_corpus)  # store to disk, for later use
    return


def get_doc_topic_dist(model, corpus, kwords=False):
    
    '''
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    '''
    top_dist =[]
    keys = []

    for d in corpus:
        tmp = {i:0 for i in range(num_topics)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [array(vals)]
        if kwords:
            keys += [array(vals).argmax()]

    return array(top_dist), keys


def use_lda_model(documents=[],model='',trained_dictionary='',verbose=True):
    global MODEL_FILENAME
    #> Again, extend https://www.kaggle.com/ykhorramz/lda-and-t-sne-interactive-visualization

    #NOTES:
    #- allow user to pass reusable model/dictionary
    
    
    if not model:
        if not os.path.exists(MODEL_FILENAME):
            print ("Training new model: "+MODEL_FILENAME)
            train_model_basic()
    
        model = LdaModel.load(MODEL_FILENAME)
        
    if not trained_dictionary:
        #Use trained dictionary to lookup bow
        trained_dictionary = corpora.Dictionary.load(TOPIC_DICTIONARY_FILENAME)
    print ("Trained dict length: "+str(len(trained_dictionary)))
    
    if not documents:
        print ("**USING DEFAULT DEMO DOCS...")
        documents=['The world bank and the world bank women economic market']
        documents+= [
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 ]
    
    #Combine# documents=[" ".join(documents)]
    
    
    
    #1/  Use trained dictionary to lookup bow and return standard text
    documents,other_texts,other_corpus,common_dictionary=get_corpora(documents=documents,common_dictionary=trained_dictionary,verbose=False)

    #2/  Docs vectors to topic vectors
    topic_vectors = model[other_corpus]
    
    #3/  Review documents
    doc_topics=[]
    for i,document in enumerate(documents):
        if verbose:
            print ("Given document: "+str(document))
            print ("Given tokenized: "+str(other_texts[i]))
            print ("Using stemmed: "+str(other_corpus[i]))
        
        #/ transform topic into top list
        topic_matches=[]
        for tnum,percent in topic_vectors[i]:topic_matches+=[(tnum,percent)]
        topic_matches.sort(key=lambda x:x[1],reverse=True)
        
        #/ Iter top topic matches (break at 0)
        tnum=-1;percent=0;topic_lable=''
        for tnum,percent in topic_matches:
            topic_label=describe_topic(model,trained_dictionary,tnum)
            if verbose:
                print ("Topic #"+str(tnum)+" match "+str(percent)+"%  >topic label: "+topic_label)
            doc_topics+=[(tnum,percent,topic_label)]
            break
        if verbose: print

    if 'review_topic_distribution_similarity_content' in '':
        lda_corpus1 = model[corpus1]
        top_dist1, _ = get_doc_topic_dist(model, lda_corpus1)
        #> possible cosine_sim...https://www.kaggle.com/ykhorramz/lda-and-t-sne-interactive-visualization

    print ("**assumption:  Topic label is first 3 most salient terms of topic cluster.")
    return doc_topics,model,trained_dictionary #For reuse

def describe_topic(model,common_dictionary,tnum):
    #vocab like get_topic_terms#  topic_info=model.print_topic(tnum,topn=10)
    dd=model.show_topic(tnum)
    terms=[]
    for id,percent in dd:
        term=common_dictionary[int(id)]
        terms+=[term]
    topic_label=" ".join(terms[:3])
    return topic_label

def doc2topics(document,lda=''):
    old=aa
    #Allow lda model persistence (makes easier but not super clean)
    global MODEL_FILENAME
    lda = LdaModel.load(MODEL_FILENAME)
    documents,other_texts,other_corpus=get_corpora(documents=[document])
    vector = lda[other_corpus[0]]
    return vector,lda #[(tnum,%),(tnum2,%)]

def visualize_topic_model():
    #pyldaviz
    #- inotebook option
    #- built-in webserver option
    print ("--->  Running visualization")

    if False: #via notebook
        pyLDAvis.enable_notebook()
        
    if not os.path.exists(MODEL_FILENAME):
        print ("Model not trained for visualization.  Training new model...")
        train_model_basic()

    print "Loading pre-trained LDA model data..."
    lda = models.ldamodel.LdaModel.load(MODEL_FILENAME)
    dictionary = corpora.Dictionary.load(TOPIC_DICTIONARY_FILENAME)
    corpus = corpora.MmCorpus(TOPIC_CORPUS_FILENAME)

    vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    
    print ("Launch pyLDAvis via webserver show... (there are inotebook options)")
    pyLDAvis.show(vis_data) #Starts web server
    

    return

def demo_topic_classifier():
    documents=['The world bank and the world bank women economic market']
    doc_topics,reuseable_model,reuseable_dict=use_lda_model(documents=documents,model='',trained_dictionary='')
    
    for i,doc in enumerate(documents):
        print ("DOC: "+str(doc))
        print (" topic info: "+str(doc_topics))
    return

if __name__=='__main__':

    branches=['use_lda_model']           #DEV


    branches=['train_model_basic']       #TRAIN MODEL

    branches=['demo_topic_classifier']   #USAGE EXAMPLE

    branches+=['visualize_topic_model']  #VISUALIZE

    for b in branches:
        globals()[b]()















