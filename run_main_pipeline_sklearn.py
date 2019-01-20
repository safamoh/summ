from __future__ import division
import pickle
from joblib import dump, load
import numpy as np
import random
from itertools import izip

from duc_reader import files2sentences
from duc_reader import tokenize_sentences
from duc_reader import get_query
from duc_reader import TEMP_DATA_PATH
from duc_reader import get_sim_matrix_path
from duc_reader import get_list_of_all_topics

from performance import Performance_Tracker 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

Perf=Performance_Tracker()


def print_sims(sims,sentences,max_lines=3):
    print ("-----> PRINT SIMS")
    print ("*skip zero cosims")

    options=[]
    options=['print_sims']

    if 'print_sims' in options:
        i=0
        j=0
        for item in list(enumerate(sims)):
            i+=1
            #            if i>0:break
            sent_num1=item[0]
            sent_text1=sentences[sent_num1].strip()
            for sent_num2,cosim_value in enumerate(item[1]):
                sent_text2=sentences[sent_num2].strip()
                if not sent_text1 or not sent_text2:continue
                
                if cosim_value==0:continue 

                j+=1
                idx="("+str(sent_num1)+","+str(sent_num2)+")"
                cosim_str="%.9f" % cosim_value
                if True and j<max_lines:
                    print ("AT: "+str(idx)+" sim: "+str(cosim_str))
                    print ("  for sent1: >"+str(sentences[sent_num1])+"<")
                    print ("   vs sent2: >"+str(sentences[sent_num2])+"<")
                    
                    if sentences[sent_num1]==sentences[sent_num2] and float(cosim_value)<0.9:
                        print ("Low sim on similar sentences stop")
                        a=hard_stop_check_logic
                if j>max_lines:break
    return


def cosine_sim_algs(matrix):
    #Note:  2005 46k sentences required 16GB temp file (likely same amount of ram)
    
    option=['via_linear_kernel']
    option=['via_chunks_to_file_tbd']
    option=['via_batches_mem_error']
    option=['std_mem_error']

    if 'std_mem_error' in option:
        print ("[debug] calculating similarity matrix...")
        sims=cosine_similarity(matrix,matrix)  #,dense_output=True)
        return sims
    
    if 'via_batches_mem_error' in option:
        #https://stackoverflow.com/questions/40900608/cosine-similarity-on-large-sparse-matrix-with-numpy
#        def cosine_similarity_n_space(m1, m2, batch_size=100):
        def cosine_similarity_n_space(m1, m2, batch_size=16):
            assert m1.shape[1] == m2.shape[1]
            print ("SHAPE: "+str(m1.shape[0])+" by "+str(m2.shape[0])) #2005 dataset gives 46kx46k
            ret = np.ndarray((m1.shape[0], m2.shape[0]))  ## MEM ERR

            for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
                start = row_i * batch_size
                end = min([(row_i + 1) * batch_size, m1.shape[0]])
                if end <= start:
                    break # cause I'm too lazy to elegantly handle edge cases
                rows = m1[start: end]
                sim = cosine_similarity(rows, m2) # rows is O(1) size
                ret[start: end] = sim
            return ret
    
        return cosine_similarity_n_space(matrix, matrix)
    
    if 'via_chunks_to_file_tbd' in option:
        chunk_size = 500 
        matrix_len = matrix.shape[0] # Not sparse numpy.ndarray

        def similarity_cosine_by_chunk(matrix,start, end):
            if end > matrix_len:
                end = matrix_len
            return cosine_similarity(X=matrix[start:end], Y=matrix) # scikit-learn function

        similarities = []
        for chunk_start in xrange(0, matrix_len, chunk_size):
            cosine_similarity_chunk = similarity_cosine_by_chunk(matrix,chunk_start, chunk_start+chunk_size)
            # Handle cosine_similarity_chunk  ( Write it to file_timestamp and close the file )
            # Do not open the same file again or you may end up with out of memory after few chunks 
            similarities.extend(cosine_similarity_chunk)
        similarities = np.array(similarities)
        return similarities
            
    if 'via_linear_kernel' in option:
        #https://stackoverflow.com/questions/46435220/calculating-similarity-between-tfidf-matrix-and-predicted-vector-causes-memory-o
    
        #This is where the memory blows up
        invec=matrix
        X=matrix

        batchsize = 1024 #mem error
        batchsize = 512
        batchsize = 256
        batchsize = 16

        similarities = []
        for i in range(0, X.shape[0], batchsize):
            similarities.extend(linear_kernel(invec, X[i:min(i+batchsize, X.shape[0])]).flatten())
        similarities = np.array(similarities)
        #similarities_orig = linear_kernel(invec, X)
        #print((similarities == similarities_orig).all())
        return similarities


class Custom_Vectorizer():
    #Follow sklearn interface 
    def __init__(self):
        #self.vectorizer_path=TEMP_DATA_PATH+'vectorizer_model_'+local_topic_id+'.mm'
        self.vectorizer_path=TEMP_DATA_PATH+'vectorizer_model.joblib'
        return
    def initialize(self):
        self.vectorizer=TfidfVectorizer()
        return
    def fit(self,input_docs):
        #'Trains' vectorizer
        return self.vectorizer.fit(input_docs)
    def fit_transform(self,input_docs):
        #'Trains' vectorizer + returns transform
        return self.vectorizer.fit_transform(input_docs)
    def transform(self,input_docs):
        #'Uses trained vectorizers'
        return self.vectorizer.transform(input_docs)
    def save(self):
        dump(self.vectorizer, self.vectorizer_path)
        return
    def load(self):
        self.vectorizer=load(self.vectorizer_path)
        return

def run_pipeline_sklearn_create_vectorizer():
    #> use all docs for vectorizer
    
    
    #1/  Get sentences
    documents,sentences,sentences_topics=files2sentences(limit_topic='')
    #Add all query sentences
    for topic_id in get_list_of_all_topics():
        query_sentence=get_query(topic_id)
        sentences.insert(0,query_sentence)
        sentences_topics.insert(0,topic_id)
        
    
    #2/  Normalize documents (prior to tf-idf)
    print ("[debug] normalizing docs...")
    input_docs=[]
    for doc in documents:
        for words in tokenize_sentences([doc]):
            input_docs+=[" ".join(words)]
        
    #3/  Create vectorizer
    print ("[debug] creating tfidf vectorizer...")

    #vectorizer = TfidfVectorizer()
    vectorizer = Custom_Vectorizer()
    vectorizer.initialize()

    vectorizer.fit(input_docs)
    #tfidf_matrix = vectorizer.fit_transform(input_docs)
    
    print ("[debug] saving vectorizer...")
    vectorizer.save()
    
 
    return

def run_pipeline_sklearn_create_sims(topic_id):
    #0/  Load vectorizer -- trained above

    print ("Loading vectorizer...")
    vectorizer=Custom_Vectorizer()
    vectorizer.initialize()
    vectorizer.load()
    

    #1/  Get sentences
    documents,sentences,sentences_topics=files2sentences(limit_topic=topic_id)

    #Add all query sentences
    query_sentence=get_query(topic_id)
    sentences.insert(0,query_sentence)
    sentences_topics.insert(0,topic_id)
        
    
    #2/  Normalize sentences (stemm etc)
    sentences_stemmed=[]
    for sentence_words_list in tokenize_sentences(sentences):
        sentence=" ".join(sentence_words_list)
        sentences_stemmed+=[sentence]
        

    #** note: similarities at topic level (otherwise too large)
    #3/  Transform sentences
    tfidf_sentences=vectorizer.transform(sentences_stemmed)
    
    print ("[debug] calculating similarity matrix...")
    print ("Sentence count: "+str(len(sentences_stemmed)))
    print ("Sparse matrix: "+str(tfidf_sentences.shape))
    sims=cosine_sim_algs(tfidf_sentences)

    #6/  Save
    print ("Save sims to: "+str(get_sim_matrix_path(topic_id)))
    np.save(get_sim_matrix_path(topic_id),sims)
            
    #7/  Print option
    print_sims(sims,sentences,max_lines=2)

    return


if __name__=='__main__':
    
    branches=['run_pipeline']

    for b in branches:
        globals()[b]()











