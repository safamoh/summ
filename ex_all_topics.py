import os
import codecs
from run_main_pipeline import run_pipeline
from run_graph_pipeline import run_random_walk_on_graph
from run_graph_pipeline import run_clustering_on_graph
from run_graph_pipeline import select_top_cos_sims
from run_graph_pipeline import do_selection_by_weight
from run_graph_pipeline import do_selection
from run_graph_pipeline import do_selection_by_round_robin
from duc_reader import get_list_of_all_topics
from duc_reader import TEMP_DATA_PATH


from run_graph_pipeline import do1_select_query_cluster
from run_graph_pipeline import do2_local_walk
from run_graph_pipeline import do3_avg_cosims
from run_graph_pipeline import do4_median_weight
from run_graph_pipeline import do5_markov_clustering
from run_graph_pipeline import do6_two_scores
from run_graph_pipeline import do6_two_scores_1
from run_graph_pipeline import do6_two_scores_2
from run_graph_pipeline import do7_sum_nodes



output_directory=TEMP_DATA_PATH+"/Top_summary"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

def run_exercise():
    global output_directory
    print ("For each topic, create sim matrix and do top n")
    
    
    
    branch=['create_sim_matrix']  #Must be run once

#    branch+=['do_random_walk']
#    branch+=['select_top_cos_sims']
#    branch+=['do_selection_multiple_cluster_algs']
#    branch+=['select_by_cluster_weight_factor']
#    branch+=['do_selection_by_round_robin']
#    branch+=['do_selection_by_round_robin']
    branch+=['experiments']

    

    all_topics=get_list_of_all_topics()
    print ("Processing topics: "+str(all_topics))
    print (str(len(all_topics))+" topics found.")
    
    for topic_id in all_topics:

        if 'create_sim_matrix' in branch:
            print ("-----------> Creating sim matrix for topic: "+str(topic_id))
            run_pipeline(use_specific_topic=topic_id)
    

#==========================================================================================
        if 'do_random_walk' in branch:
            top_n=10
            top_n_output=output_directory+"/random_walk/"+str(topic_id)+".txt"
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
#==========================================================================================          
        if 'select_top_cos_sims' in branch:
            top_n_output=output_directory+"/cos_sim/"+str(topic_id)+".txt"
            print ("Calc top cos sims for topic: "+str(topic_id)+"...")
            fp=codecs.open(top_n_output,'w',encoding='utf-8')
            for sentence in select_top_cos_sims(topic_id=topic_id,top_n=10):
                fp.write(sentence+"\n")
            fp.close()
#==========================================================================================
        if 'select_by_cluster_weight_factor' in branch:
            sub_branches=['fast_greedy','leading_eigenvector','walktrap']

            for sub_branch in sub_branches:
                out_report_dir=output_directory+"/do_selection_by_weight/"+sub_branch
                out_report_file=out_report_dir+"/"+str(topic_id)+".txt"
                if not os.path.exists(output_directory+"/do_selection_by_weight"):
                    os.mkdir(output_directory+"/do_selection_by_weight")
                if not os.path.exists(out_report_dir):
                    os.mkdir(out_report_dir)

                print ("For topic: "+str(topic_id)+" doing clustering: "+str(sub_branch)+" and selection report to: "+str(out_report_file))
                g,clusters,cluster_weights,query_sentence,query_index=run_clustering_on_graph(topic_id=topic_id,method=sub_branch)
                print ("Doing selection")
                fp=codecs.open(out_report_file,'w',encoding='utf-8')
                for sentence in do_selection_by_weight(g,clusters,cluster_weights,query_sentence,query_index):
                    fp.write(sentence+"\n")
                fp.close()
 #==========================================================================================               
        if 'do_selection_multiple_cluster_algs' in branch:
            sub_branches=['fast_greedy','leading_eigenvector','walktrap']

            for sub_branch in sub_branches:
                out_report_dir=output_directory+"/do_selection/"+sub_branch
                out_report_file=out_report_dir+"/"+str(topic_id)+".txt"
                if not os.path.exists(output_directory+"/do_selection"):
                    os.mkdir(output_directory+"/do_selection")
                if not os.path.exists(out_report_dir):
                    os.mkdir(out_report_dir)

                print ("For topic: "+str(topic_id)+" doing clustering: "+str(sub_branch)+" and selection report to: "+str(out_report_file))
                g,clusters,cluster_weights,query_sentence,query_index=run_clustering_on_graph(topic_id=topic_id,method=sub_branch)
                print ("Doing selection")
                fp=codecs.open(out_report_file,'w',encoding='utf-8')
                for sentence in do_selection(g,clusters,cluster_weights,query_sentence,query_index):
                    fp.write(sentence+"\n")
                fp.close()
            
#==========================================================================================          
        if 'do_selection_by_round_robin' in branch:
            sub_branches=['fast_greedy','leading_eigenvector','walktrap']

            for sub_branch in sub_branches:
                out_report_dir=output_directory+"/round_robin/"+sub_branch
                out_report_file=out_report_dir+"/"+str(topic_id)+".txt"
                if not os.path.exists(output_directory+"/round_robin"):
                    os.mkdir(output_directory+"/round_robin")
                if not os.path.exists(out_report_dir):
                    os.mkdir(out_report_dir)

                print ("For topic: "+str(topic_id)+" doing clustering: "+str(sub_branch)+" and selection report to: "+str(out_report_file))
                g,clusters,cluster_weights,query_sentence,query_index=run_clustering_on_graph(topic_id=topic_id,method=sub_branch)
                print ("Doing selection")
                fp=codecs.open(out_report_file,'w',encoding='utf-8')
                for sentence in do_selection_by_round_robin(g,clusters,cluster_weights,query_sentence,query_index,target_sentences=10):
                    fp.write(sentence+"\n")
                fp.close()

#==========================================================================================
        if 'experiments' in branch:
            exs=[]
            #exs+=['do1_select_query_cluster']
            #exs+=['do2_local_walk']
            #exs+=['do3_avg_cosims']
            #exs+=['do4_median_weight']
            #exs+=['do5_markov_clustering']
            #exs+=['do6_two_scores']
            #exs+=['do6_two_scores_1']
            #exs+=['do6_two_scores_2']
            exs+=['do7_sum_nodes']
            
            for experiment in exs:
                ex_name="ex_"+experiment
                if experiment=='do5_markov_clustering':
                    sub_branches=['markov']
                else:
                    sub_branches=['leading_eigenvector']
    
                for sub_branch in sub_branches:
                    if exs=="do7_sum_nodes":
                        out_report_dir=output_directory+"/"+ex_name
                    else:
                        out_report_dir=output_directory+"/"+ex_name+"/"+sub_branch

                    out_report_file=out_report_dir+"/"+str(topic_id)+".txt"
                    if not os.path.exists(output_directory+"/"+ex_name):
                        os.mkdir(output_directory+"/"+ex_name)
                    if not os.path.exists(out_report_dir):
                        os.mkdir(out_report_dir)
    
                    print ("Experiment: "+str(experiment)+" For topic: "+str(topic_id)+" doing clustering: "+str(sub_branch)+" and selection report to: "+str(out_report_file))
                    g,clusters,cluster_weights,query_sentence,query_index=run_clustering_on_graph(topic_id=topic_id,method=sub_branch,experiment=experiment)
                    print ("Doing selection")
                    fp=codecs.open(out_report_file,'w',encoding='utf-8')
                    the_function=globals()[experiment]
                    for sentence in the_function(g,clusters,cluster_weights,query_sentence,query_index,topic_id=topic_id):
                        fp.write(sentence+"\n")
                    fp.close()
#                break #at first experiment
        print ("Done topic: "+str(topic_id))
#        break #break #at first topic
#                    break #at first cluster
#                break #at first experiment
#        break #break #at first topic


    print ("Done run_exercise...")
    return
#==========================================================================================
if __name__=='__main__':
    branches=['run_exercise']
    for b in branches:
        globals()[b]()
        
        
        
        
























