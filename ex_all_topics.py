import os
import codecs
from run_main_pipeline import run_pipeline
from run_graph_pipeline import run_random_walk_on_graph
from run_graph_pipeline import select_top_cos_sims
from duc_reader import get_list_of_all_topics
from duc_reader import TEMP_DATA_PATH

#0v1#  Oct 1, 2018

#Request:
#Run for each topic

output_directory=TEMP_DATA_PATH+"/Top_summary"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

def run_exercise():
    global output_directory

    print ("For each topic, create sim matrix and do top n")

    branch=['create_sim_matrix']

    #branch+=['do_random_walk']
    branch+=['do_random_walk']
    branch+=['select_top_cos_sims']
    
    if 'do_random_walk' in branch:
        output_directory+="/random_walk"
    elif 'select_top_cos_sims' in branch:
        output_directory+="/cos_sim"
    

    all_topics=get_list_of_all_topics()
#    all_topics=['d301i']
    print ("Processing topics: "+str(all_topics))
    print (str(len(all_topics))+" topics found.")
    
    for topic_id in all_topics:
        top_n=10
        top_n_output=output_directory+"/"+str(topic_id)+".txt"

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
            
        if 'select_top_cos_sims' in branch:
            print ("Calc top cos sims for topic: "+str(topic_id)+"...")
            fp=codecs.open(top_n_output,'w',encoding='utf-8')
            for sentence in select_top_cos_sims(topic_id=topic_id):
                fp.write(sentence+"\n")
            fp.close()
            
#        break

    print ("Done run_exercise...")
    return

if __name__=='__main__':
    branches=['run_exercise']
    for b in branches:
        globals()[b]()
        
        
        
        
























