import os
import sys
import re
import codecs

import nltk
import nltk.data
from nltk.stem.porter import PorterStemmer
from lxml import etree #pip install lxml
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
import numpy as np
import igraph


ENC='utf-8'
sys.stdout = codecs.getwriter(ENC)(sys.stdout) #Allow special characters to be printed
sys.stderr = codecs.getwriter(ENC)(sys.stderr)
LOCAL_DIR = os.path.abspath(os.path.dirname(__file__))+"/"

pStemmer = PorterStemmer() #For vectorizing


FILES_TO_PROCESS=10000000
VERSION=3


# SET BASE FOLDER OF INPUT FILES:
DOCS_PATH=LOCAL_DIR+"../data/2005/DUC2005_Summarization_Documents/duc2005_docs.tar/duc2005_docs"
TOPIC_FILENAME=LOCAL_DIR+"../data/2005/duc2005_topics.sgml"
TEMP_DATA_PATH=LOCAL_DIR+"../data/"

if not os.path.exists(DOCS_PATH):
    DOCS_PATH="/Users/safaalballaa/Desktop/duc/2005/DUC2005_Summarization_Documents"
    TOPIC_FILENAME="/Users/safaalballaa/Desktop/duc/2005/duc2005_topics.sgml"
    TEMP_DATA_PATH="/Users/safaalballaa/Desktop/resulted_files/"
    if not os.path.exists(DOCS_PATH):
        print ("Input directory invalid: "+str(DOCS_PATH))
        hard_stop=bad_dir_inputs
if not os.path.exists(TEMP_DATA_PATH):
    print ("Error in configuration of temp directory: "+str(TEMP_DATA_PATH))
    



#############################################
#  GLOBAL CONFIGS
#############################################
#
TOPIC_ID='d301i'       
#TOPIC_ID='d307b'
#TOPIC_ID='d311i'
#TOPIC_ID='d313e'

LIMIT_TOPICS=True      #Look at subset of data
#
#
#############################################

def get_sim_matrix_path(local_topic_id,cosim_topic_signatures=False):
    global TEMP_DATA_PATH
    if cosim_topic_signatures:
        smp=TEMP_DATA_PATH+"_"+local_topic_id+"_TSIG_sim_matrix.npy"
    else:
        smp=TEMP_DATA_PATH+"_"+local_topic_id+"_sim_matrix.npy"
    return smp

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
                    
def clean_sentence(sentence):
    sentence=sentence.strip()
    sentence=re.sub(r'([\. ]+)',r'\1',sentence) #remove repeated spaces and periods
    if len(sentence)==1:sentence=''
    return sentence

def filter_out_sentence(sentence):
    filter=False
    ## Filter:  Sentence must have 1 alpha numeric
    if not re.search(r'\w',sentence):filter=True
    
    ## Filter:  Remove sentence with 2 tokens or less.. (Oct 10th)
    token_count=len(re.findall(r'[ ]+',sentence)) #Count tokens
    if token_count<2:
        filter=True
        
    ## Filter:  Remove sentence which is part of table (Oct 30th)
    if re.search(r'-----------.* .*----------',sentence): #Table
#        print ("[query_indexdebug] filter table: "+str(sentence))
        filter=True

    return filter
    
def files2sentences(limit_topic='',limit=0):
    #>update to grab DUC id
    global DOCS_PATH,ENC,FILES_TO_PROCESS
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
   # print ("[debug] using training documents at: "+str(DOCS_PATH))

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
        try:
            doc_text=xml2text(filename) #Catch enabled
            documents+=[doc_text]
        except:
            print ("[error] could not process input xml: "+str(filename))

   # print ("[debug] Tokenizing input sentences...")
    # flatten all documents into list of sentences
    sentences=[]
    sentence_topics=[]
    for i,document in enumerate(documents):
        for sentence in sent_detector.tokenize(document):
            sentence=clean_sentence(sentence)
            if not filter_out_sentence(sentence):
                sentences+=[sentence]
                sentence_topics+=[document_topics[i]]
    #print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents.")
    return documents,sentences,sentence_topics

def get_list_of_all_topics():
    custom_topic_list=[]
    documents,sentences,sentence_topics=files2sentences()
    unique_topics=list(set(sentence_topics))
    for topic_id in unique_topics:
        if re.search(r'^d\d',topic_id):
            custom_topic_list+=[topic_id]
    return custom_topic_list

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

    #invalid xml#  tree = etree.parse(TOPIC_FILENAME)
    #root = tree.getroot()
    with open(TOPIC_FILENAME, 'r') as xml:
        xmlstring = ''.join(xml.readlines())
        xmlstring="<data>"+xmlstring+"</data>" #sgml has no wrap
        root = etree.fromstring(xmlstring)
        tree = etree.ElementTree(root)
        for text in tree.iter():
            if text.tag == 'num':
                last_topic=text.text.strip()
            if text.tag=='narr':
                blob=text.text
                if topic_id==last_topic:
                    found=True
                    break
    if not found:blob=''
    blob=re.sub(r'\n',' ',blob)
    #print ("TOPIC FILENAME: "+str(TOPIC_FILENAME))
    print ("FOR TOPIC: "+str(topic_id)+" got query: "+str(blob))
    return blob

#######################################################################


if __name__=='__main__':
    branches=[]
    for b in branches:
        globals()[b]()





















