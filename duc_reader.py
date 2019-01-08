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

#0v2# Dec 27, 2018  Added support for 2006, 2007 data input

ENC='utf-8'
sys.stdout = codecs.getwriter(ENC)(sys.stdout) #Allow special characters to be printed
sys.stderr = codecs.getwriter(ENC)(sys.stderr)
LOCAL_DIR = os.path.abspath(os.path.dirname(__file__))+"/"

pStemmer = PorterStemmer() #For vectorizing


FILES_TO_PROCESS=10000000
VERSION=3

##########################
#  GLOBAL DATA SOURCES
##########################
#DOCS_SOURCE='2005'
DOCS_SOURCE='2006'
#DOCS_SOURCE='2007'

if DOCS_SOURCE=='2005': #org default
    TEMP_DATA_PATH=LOCAL_DIR+"../data/"

    # SET BASE FOLDER OF INPUT FILES:
    DOCS_PATH=LOCAL_DIR+"../data/2005/DUC2005_Summarization_Documents/duc2005_docs.tar/duc2005_docs"
    TOPIC_FILENAME=LOCAL_DIR+"../data/2005/duc2005_topics.sgml"
    
    if not os.path.exists(DOCS_PATH):
        TEMP_DATA_PATH="/Users/safaalballaa/Desktop/resulted_files/"
        DOCS_PATH="/Users/safaalballaa/Desktop/duc/2005/DUC2005_Summarization_Documents"
        TOPIC_FILENAME="/Users/safaalballaa/Desktop/duc/2005/duc2005_topics.sgml"
        if not os.path.exists(DOCS_PATH):
            print ("Input directory invalid: "+str(DOCS_PATH))
            hard_stop=bad_dir_inputs
elif DOCS_SOURCE=='2006':
    # SET BASE FOLDER OF INPUT FILES:
    TEMP_DATA_PATH=LOCAL_DIR+"../data/models_2006" #** NEW

    DOCS_PATH=LOCAL_DIR+"../data/2006/DUC2006_Summarization_Documents/duc2006_docs"
    TOPIC_FILENAME=LOCAL_DIR+"../data/2006/duc2006_topics.sgml"
    
    if not os.path.exists(DOCS_PATH):
        TEMP_DATA_PATH="/Users/safaalballaa/Desktop/resulted_files/2006"
        DOCS_PATH="/Users/safaalballaa/Desktop/duc/2006/DUC2006_Summarization_Documents/duc2006_docs"
        TOPIC_FILENAME="/Users/safaalballaa/Desktop/duc/2006/duc2006_topics.sgml"
elif DOCS_SOURCE=='2007':
    TEMP_DATA_PATH=LOCAL_DIR+"../data/models_2007" #** NEW

    DOCS_PATH=LOCAL_DIR+"../data/2007/DUC2007_Summarization_Documents/duc2007_testdocs/main"
    TOPIC_FILENAME=LOCAL_DIR+"../data/2007/duc2007_topics.sgml"
    
    if not os.path.exists(DOCS_PATH):
        TEMP_DATA_PATH="/Users/safaalballaa/Desktop/resulted_files/2007"
        DOCS_PATH="/Users/safaalballaa/Desktop/duc/2007/DUC2007_Summarization_Documents/duc2007_testdocs/main"
        TOPIC_FILENAME="/Users/safaalballaa/Desktop/duc/2007/duc2007_topics.sgml"


## VALIDATE DATA SOURCES
if not os.path.exists(DOCS_PATH):
    print ("Input directory invalid: "+str(DOCS_PATH))
    hard_stop=bad_dir_inputs
try: os.mkdir(TEMP_DATA_PATH)
except:pass
        



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
    #0v2#  Dec 27, 2018
    #>  2006 has TEXT nested under BODY tag.
    #>  If no TEXT found then use <P> tags

    #sum_utilities.py
    blob=''
    p_blob='' #Just collect paragraph details
    with open(xml_filename, 'r') as xml:
        xmlstring = ''.join(xml.readlines())
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xmlstring, parser=parser)
        tree = etree.ElementTree(root)
        for node in root:
            if node.tag == text_tag:
                content=etree.tostring(node)
                content=re.sub(r'\<.{0,1}TEXT\>','',content)
                content=re.sub(r'\<.{0,1}P\>','',content)
                blob+=content

            for node2 in node:
                if node.tag == 'BODY': #Process nested tag
                    if node2.tag == text_tag:
                        content=etree.tostring(node2)
                        content=re.sub(r'\<.{0,1}TEXT\>','',content)
                        content=re.sub(r'\<.{0,1}P\>','',content)
                        blob+=content

                else: #Process text where NO BODY
                    if node2.tag == 'P':
                        content=etree.tostring(node2)
                        content=re.sub(r'\<.{0,1}TEXT\>','',content)
                        content=re.sub(r'\<.{0,1}P\>','',content)
                        p_blob+=content

            
    blob=re.sub(r'\n',' ',blob)
    
    if not blob.strip():
        print ("Using non-body paragraph text: "+xml_filename)
        blob=p_blob
    #print "FOR FILE: "+str(xml_filename)
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
    
def files2sentences(limit_topic='',limit=0,verbose=True):
    #>update to grab DUC id
    global DOCS_PATH,ENC,FILES_TO_PROCESS,DOCS_SOURCE
    
    #DOCS_SOURCE==2005, 2006, 
    
    if verbose:
        print ("SOURCING SENTENCES FOR YEAR: "+str(DOCS_SOURCE))
        print ("SOURCING SENTENCES FROM: "+str(DOCS_PATH))
        if limit_topic:
            print ("**LIMITING TOPIC TO: "+str(limit_topic))
    
    
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
   # print ("[debug] using training documents at: "+str(DOCS_PATH))

    # get list of filenames in the directory
    filenames=[]
    document_topics=[]
    c=0
    for path in walk_directory([DOCS_PATH]):
        c+=1
        if c==1 and verbose:
            print ("Sample first file: "+str(path))

        temp=re.sub(r'.*[\\\/]','',path)
        topic_duc=re.split(r'\/',path)[-2]
        if not re.search(r'[dD]\d+\w',topic_duc): #2005 is d2322, 2006 is D05555E
            print ("SKIPPING TOPIC ID: "+str(topic_duc))
            topic_doc=''
        
        
        #Don't open if topic is restrictive
        if limit_topic and not limit_topic.lower()==topic_duc:
            print ("Skipping unrelated topic: "+path)
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
            
            if not doc_text.strip():
                print ("No content found in: "+str(filename))
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
#    print ("Loaded "+str(len(sentences))+" sentences from "+str(len(documents))+" documents. topic count: "+str(len(sentence_topics)))
    return documents,sentences,sentence_topics

def get_list_of_all_topics():
    custom_topic_list=[]
    documents,sentences,sentence_topics=files2sentences()
    unique_topics=list(set(sentence_topics))
    for topic_id in unique_topics:
        if re.search(r'[dD]\d+\w',topic_id):
            custom_topic_list+=[topic_id]
        else:
            print ("[warning] skipping topic: "+str(topic_id))
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

def test_different_sources():
    source=['']
    print ("Change top years as required")
    documents,sentences,sentence_topics=files2sentences()
    print ("FOUND SENTENCE COUNT: "+str(len(sentences)))
    print ("FOUND TOPIC COUNT: "+str(len(list(set(sentence_topics)))))
    return


if __name__=='__main__':
    branches=['test_different_sources']
    for b in branches:
        globals()[b]()





















