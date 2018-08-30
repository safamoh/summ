import os
import sys
import re
import nltk
import nltk.data
import codecs
from lxml import etree #pip install lxml
from nltk.stem.porter import PorterStemmer
from lxml import html
import gensim
from gensim.matutils import jaccard, cossim, dense2vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import models
from gensim import corpora
from gensim import similarities
import string


#github

ENC='utf-8'
sys.stdout = codecs.getwriter(ENC)(sys.stdout) #Allow special characters to be printed
sys.stderr = codecs.getwriter(ENC)(sys.stderr)
LOCAL_DIR = os.path.abspath(os.path.dirname(__file__))+"/"

pStemmer = PorterStemmer() #For vectorizing

#0v1# JC Aug 27, 2018  Run pipeline setup


FILES_TO_PROCESS=10000000
VERSION=2


# SET BASE FOLDER OF INPUT FILES:
DOCS_PATH=LOCAL_DIR+"data/2005/DUC2005_Summarization_Documents/duc2005_docs.tar/duc2005_docs"
TOPIC_FILENAME=LOCAL_DIR+"data/2005/duc2005_topics.sgml"

if not os.path.exists(DOCS_PATH):
    DOCS_PATH="/Users/safaalballaa/Desktop/duc/2005/DUC2005_Summarization_Documents"
    TOPIC_FILENAME="/Users/safaalballaa/Desktop/duc/2005/duc2005_topics.sgml"
    if not os.path.exists(DOCS_PATH):
        print ("Input directory invalid: "+str(DOCS_PATH))
        hard_stop=bad_dir_inputs



def walk_directory(folders,include_dir=False):
    if not isinstance(folders,list):folders=[folders]
    for folder in folders:
        for dirname, dirnames, filenames in os.walk(folder):
            if include_dir:
                yield dirname
            for filename in filenames:
                path=os.path.join(dirname,filename)
                if re.search(r'\.swp$',path):continue #No temp files
                if re.search(r'~$',path):continue #No temp files
                if os.path.isfile(path):
                    path=re.sub(r'\\','/',path)
                    yield path
                    
def xml2text(xml_filename,text_tag='TEXT'):
    #sum_utilities.py
    blob=''
    with open(xml_filename, 'r') as xml:
        xmlstring = ''.join(xml.readlines())
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xmlstring, parser=parser)
        tree = etree.ElementTree(root)
        for text in root:
            if text.tag == text_tag:
                content=etree.tostring(text)
                content=re.sub(r'\<.{0,1}TEXT\>','',content)
                content=re.sub(r'\<.{0,1}P\>','',content)
                blob+=content
    blob=re.sub(r'\n',' ',blob)
    #print "FOR FILE: "+str(xml_filename)
    #print ("BLOB: "+str(blob))
    return blob
                    
    
def files2sentences(limit_topic='',limit=0):
    #>update to grab DUC id
    global DOCS_PATH,ENC,FILES_TO_PROCESS
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # get list of filenames in the directory
    filenames=[]
    document_topics=[]
    for path in walk_directory([DOCS_PATH]):

        temp=re.sub(r'.*[\\\/]','',path)
        topic_duc=re.split(r'\/',path)[-2]
        if not re.search(r'd\d+\w',topic_duc): topic_doc=''
        
        
        #Don't open if topic is restrictive
        if limit_topic and not limit_topic.lower()==topic_duc:
#            print ("Skipping unrelated topic: "+path)
            continue

        document_topics+=[topic_duc]
        filenames+=[path]
        if limit<0: #No limit
            pass
        elif len(filenames)>FILES_TO_PROCESS:break

    documents=[]
    for filename in filenames:
        doc_text=xml2text(filename)
        documents+=[doc_text]

    print ("[debug] Tokenizing input sentences...")
    # flatten all documents into list of sentences
    sentences=[]
    sentence_topics=[]
    for i,document in enumerate(documents):
        for sentence in sent_detector.tokenize(document):
            sentence.strip()
            if sentence:
                sentences+=[sentence]
                sentence_topics+=[document_topics[i]]
    return documents,sentences,sentence_topics


def tokenize_sentences(sentences):
    #> lots of ways to chunk  & clean sentences
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentencesTokens = [regex.sub('', sen.lower()).split(' ') for sen in sentences]

    for i in range(len(sentencesTokens)):
        tokens = []
        for s in sentencesTokens[i]:
            if s not in STOPWORDS:
                try:
                    tokens.append(pStemmer.stem(s))
                except UnicodeDecodeError:
                    pass
        sentencesTokens[i] = tokens
    return sentencesTokens


def get_query(topic_id):
    #sum_utilities.py
    found=False
    with open(TOPIC_FILENAME, 'r') as xml:
        xmlstring = ''.join(xml.readlines())
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xmlstring, parser=parser)
        last_topic=''
        for text in root:
            if text.tag == 'num':
                last_topic=text.text
            if text.tag=='narr':
                blob=text.text
                if topic_id==topic_id:
                    found=True
                    break
    if not found:blob=''
    blob=re.sub(r'\n',' ',blob)
    return blob

#######################################################################

def run_pipeline(verbose=True):

    #0/  Load query sentence
    vector_model='tfidf'

    topic_id='d301i' #

    query_sentence=get_query(topic_id)
    print ("Using query: "+str(query_sentence))


    #1/  LOAD
    #################################


    print ("1/  Loading sentences...")
    documents,sentences,sentences_topics=files2sentences(limit_topic=topic_id) #watch loading 2x data into mem
    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents.")
    print("---------------------------------")
    for i,sentence in enumerate(sentences):
        print ("Sample sentence.  Topic: "+str(sentences_topics[i])+": "+sentence)
        if i>2:break
        
    #Add query as V1
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,topic_id)




    #2/  Normalize corpus
    ##########################################

    ##print("---------------------------------")
    ##print("list of sentences:")
    ##print sentences
    ##print("---------------------------------")
    ##print("Tokenize sentences (After using PorterStemmer):")
    norm_sentences=tokenize_sentences(sentences)
    ##print norm_sentences
    ##print("---------------------------------")





    #STEP 3 : Index and vectorize
    #####################################################

    #We create a dictionary, an index of all unique values: <class 'gensim.corpora.dictionary.Dictionary'>
    #the Dictionary is used as an index to convert words into integers.
    dictionary = corpora.Dictionary(norm_sentences)
    ##print (dictionary)
    ##print("---------------------------------")
    ##print("Dictionary (token:id):")
    ##print(dictionary.token2id)
    ##print("---------------------------------")
    dictionary.save('doc_dict.dict') # store the dictionary, for future reference
    dictionary.save_as_text('doc_txt_dict.txt',sort_by_word=False) # SAVE the dictionary as a text file,
    #the format of doc_txt_dict.txt is: (id_1    word_1  document_frequency_1)

    #---------------------------------

    # compile corpus (vectors number of times each elements appears)
    #The "compile corpus" section actually converts each sentence into a list of integers ("integer" bag-of-words)
    #This raw_corpus is then fed into the tfidf model.
    raw_corpus = [dictionary.doc2bow(t) for t in norm_sentences]
    #Then convert tokenized documents to vectors: <type 'list'>
    print "Then convert tokenized documents to vectors: %s"% type(raw_corpus)
    #each document is a list of sentence (vectors) --> (id of the word, tf in this doc)
    ##print("raw_corpus:")
    ##print raw_corpus 
    #Save the vectorized corpus as a .mm file
    corpora.MmCorpus.serialize('doc_vectors.mm', raw_corpus) # store to disk
    print "Save the vectorized corpus as a .mm file"



    # STEP 4 : tfidf
    ###############################################
    corpus = corpora.MmCorpus('doc_vectors.mm')

    # Transform Text with TF-IDF
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    print "We initialize our TF-IDF transformation tool : %s"%type(tfidf)
   
    # corpus tf-idf
    corpus_tfidf = tfidf[corpus]
    print "We convert our vectors corpus to TF-IDF space : %s"%type(corpus_tfidf)
   



    # STEP 5 : Create similarity matrix of all files
    ###############################################
   
    index = similarities.MatrixSimilarity(tfidf[corpus])
    #print "We compute similarities from the TF-IDF corpus : %s"%type(index)
    index.save('sim_index.index')
    index = similarities.MatrixSimilarity.load('sim_index.index')
    
    sims = index[corpus_tfidf]
    #print "We get a similarity matrix for all sentences in the corpus %s"% type(sims)


    # STEP 6:  Print sims 
    ###############################################
    i=0
    for item in list(enumerate(sims)):
        i+=1
        if i>0:break
        sent_num1=item[0]
        for sent_num2,cosim_value in enumerate(item[1]):
            idx="("+str(sent_num1)+","+str(sent_num2)+")"
            cosim_str="%.9f" % cosim_value
            print ("AT: "+str(idx)+" sim: "+str(cosim_str))
            print ("  for sent1: "+str(sentences[sent_num1]))
            print ("   vs sent2: "+str(sentences[sent_num2]))

    return

    #################################################


if __name__=='__main__':
    branches=['run_pipeline']
    for b in branches:
        globals()[b]()
