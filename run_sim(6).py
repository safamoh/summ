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
import string
from igraph import Graph
from igraph import summary

#ay shay ay shay kilma wa7da

#not yet#  from run_topics import doc2topics

ENC='utf-8'
sys.stdout = codecs.getwriter(ENC)(sys.stdout) #Allow special characters to be printed
sys.stderr = codecs.getwriter(ENC)(sys.stderr)
LOCAL_DIR = os.path.abspath(os.path.dirname(__file__))+"/"

pStemmer = PorterStemmer() #For vectorizing


#0v5# JC Aug 24, 2018  Workflow 3:  Update iterator
#0v4# JC Aug 24, 2018  Workflow 3:  Add igraph, add html conversion rather then text
#0v3# JC Aug 23, 2018  Workflow 2:  Add input query, add topic ids, remove NetworkX (iGraph next)
#0v2# JC Aug 22, 2018  Workflow setup
#0v1# JC Aug 22, 2018  Infrastructure setup


##  INSTALLATION:
# http://mkelsey.com/2013/04/30/how-i-setup-virtualenv-and-virtualenvwrapper-on-my-mac/
# mkvirtualenv duc

#pip install gensim
#pip install python-louvain
#pip install nltk
#pip install lxml

#pip install python-igraph
#Binaries:
#https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph
#https://pypi.org/project/python-igraph/#files

#REFERENCE:
#http://igraph.org/python/doc/igraph.VertexSeq-class.html
#https://radimrehurek.com/gensim/models/word2vec.html


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

def create_vector_model(sentences):
    sentencesTokens=tokenize_sentences(sentences)
    vmodel = Word2Vec(sentencesTokens)
    return vmodel,sentencesTokens

def calc_sentence_similarity(sen1,sen2,vmodel,sen1a,sen2a):
    sen1 = sum([vmodel[w] for w in sen1a if w in vmodel])
    sen2 = sum([vmodel[w] for w in sen2a if w in vmodel])
    simScore = cossim(dense2vec(sen1), dense2vec(sen2))
    return simScore


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


def run_workflow2(verbose=True):
    #0/  Load query sentence
    G= Graph()

    topic_id='d301i' #

    query_sentence=get_query(topic_id)
    print ("Using query: "+str(query_sentence))

    #1/  LOAD
    print ("1/  Loading sentences...")
    documents,sentences,sentences_topics=files2sentences(limit_topic=topic_id) #watch loading 2x data into mem
    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents.")

    for i,sentence in enumerate(sentences):
        print ("Sample sentence.  Topic: "+str(sentences_topics[i])+": "+sentence)
        if i>3:break
        
    #Add query as V1
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,topic_id)
        

    #2/  Create vector model (for similarities)
    print ("Create vector model...")
    vmodel,sentencesTokens=create_vector_model(sentences)
        
    #3/  Create Graph
    G.add_vertices(sentences)

    #4/  SIMILARITY Calcs
    print "Calculating similarities..."
    for i,sentence in enumerate(sentences):
        similarity=calc_sentence_similarity(sentences[0], sentence, vmodel, sentencesTokens[0],sentencesTokens[i])
        if i<3: print ("Similarity: "+str(similarity)+" between query and: "+sentence)
        if i<3: print ("  Similarity: "+str(similarity)+" between query and: "+str(sentencesTokens[i]))

        #4/  Create Graph edges
        if i>0:
            G.add_edge(sentences[0],sentence,weight=similarity)
    
    #5/  Print graph
    summary(G)
    
    print ("Done workflow2")
    return

if __name__=='__main__':
    branches=['run_workflow2']
    for b in branches:
        globals()[b]()

























