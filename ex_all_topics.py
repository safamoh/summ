import codecs
from run_main_pipeline import run_pipeline
from run_graph_pipeline import run_random_walk_on_graph
from duc_reader import get_list_of_all_topics
from duc_reader import TEMP_DATA_PATH

#0v1#  Oct 1, 2018

#Request:
#Run for each topic


def run_exercise():
    print ("For each topic, create sim matrix and do random walk")

    branch=['create_sim_matrix']
    branch+=['do_random_walk']
    
    all_topics=get_list_of_all_topics()
#    all_topics=['d301i']
    print ("Processing topics: "+str(all_topics))
    print (str(len(all_topics))+" topics found.")
    
    for topic_id in all_topics:
        top_n=10
        top_n_output=TEMP_DATA_PATH+"top_"+str(top_n)+"_walk_scores_"+str(topic_id)+".txt"

        if 'create_sim_matrix' in branch:
            print ("-----------> Creating sim matrix for topic: "+str(topic_id))
            run_pipeline(use_specific_topic=topic_id)
    
        if 'do_random_walk' in branch:
            fp=codecs.open(top_n_output,'w',encoding='utf-8')
            c=0
            sorted_scores,query_index=run_random_walk_on_graph(topic_id)
            for idx in sorted_scores:
                if idx==query_index:continue #skip query sentence
                c+=1
                score,vertex,rank=sorted_scores[idx]
                fp.write(vertex['label']+"\n")
                if c==top_n:break
            fp.close()
            print ("Wrote to: "+top_n_output)

    print ("Done run_exercise...")
    return

if __name__=='__main__':
    branches=['run_exercise']
    for b in branches:
        globals()[b]()
        
        
        
        

