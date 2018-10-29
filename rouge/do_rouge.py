import re
import os
import sys
from pyrouge import Rouge155

sys.path.insert(0,"..")
from duc_reader import get_list_of_all_topics
from duc_reader import TEMP_DATA_PATH


LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))+"/"

is_windows = sys.platform.startswith('win')

if is_windows:
    OUTPUT_ROUGE_DIR=LOCAL_PATH+"/rouge_results"
    r = Rouge155()
else:
    OUTPUT_ROUGE_DIR="/Users/safaalballaa/Desktop/resulted_files/rouge_results"
    r = Rouge155("/Users/safaalballaa/Desktop/test/summ/rouge/ROUGE-1.5.5")

print ("Rouge install ok!")


#0v2#  JC  Oct 9, 2018  #Update paths
#0v1#  JC  Oct 2, 2018  Initial setup

#My summaries:    system
#Gold summaries:  model


def run_on_different_setups():
    system_dir = TEMP_DATA_PATH+"Top_summary"
    print ("Getting list of topics...")
    all_topics=get_list_of_all_topics()
    

    #branches=['do_random_walk']
    #branches=['select_top_cos_sims']
    #branches=['select_by_cluster_weight_factor']
    #branches=['do_selection_multiple_cluster_algs'] #do_selection original
    branches=['experiments']
    

    #option='run_individual_topics'
    option='run_all_topics_together'


#==========================================================================================
    #Special case 3 branches
    if 'do_selection_multiple_cluster_algs' in branches:
        #sub_branches=['fast_greedy']
        #sub_branches=['leading_eigenvector']
        sub_branches=['walktrap']
        for sub_branch in sub_branches:
            local_system_dir=system_dir+"/do_selection/"+sub_branch
            output_filename_base=OUTPUT_ROUGE_DIR+"/do_selection/"+sub_branch+"_"
            if 'run_individual_topics'==option:
                run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
            else:
                print ("Running rouge for all topics together")
                run_on_all_topics(system_dir=local_system_dir)
#==========================================================================================
    #Special case 3 branches
    if 'select_by_cluster_weight_factor' in branches:
        #sub_branches=['fast_greedy']
        #sub_branches=['leading_eigenvector']
        sub_branches=['walktrap']        
        for sub_branch in sub_branches:
            local_system_dir=system_dir+"/do_selection_by_weight/"+sub_branch
            output_filename_base=OUTPUT_ROUGE_DIR+"/do_selection_by_weight/"+sub_branch+"_"
            if 'run_individual_topics'==option:
                run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
            else:
                print ("Running rouge for all topics together")
                run_on_all_topics(system_dir=local_system_dir)
#==========================================================================================
    #Do standard runs
    if 'select_top_cos_sims' in branches:
        local_system_dir=system_dir+"/cos_sim"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)
#==========================================================================================

    if 'do_random_walk' in branches:
        local_system_dir=system_dir+"/random_walk"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)
#==========================================================================================
    if 'experiments' in branches:
        exs=[]
        #tbd#  exs=['do4_median_weight']
        exs+=['do1_select_query_cluster']
        exs+=['do2_local_walk']
        exs+=['do3_avg_cosims']
        exs+=['do4_median_weight']
        exs+=['do5_markov_clustering']
        exs+=['do6_two_scores']
        
        for cluster_type in sub_dirs:
            for experiment in exs:
                
                if experiment=='do5_markov_clustering':
                    sub_dirs=['markov']
                else:
                    sub_dirs=['fast_greedy','leading_eigenvector','walktrap']
                    
                ex_name="ex_"+experiment
                    
                local_system_dir=system_dir+"/"+ex_name+"/"+cluster_type
                output_filename_base=OUTPUT_ROUGE_DIR
        
                if 'run_individual_topics'==option:
                    run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
                else:
                    print ("Running rouge for all topics together")
                    run_on_all_topics(system_dir=local_system_dir)

    return


def run_one_topic_at_a_time(system_dir='',output_filename_base='',all_topics=[]):
    for topic_id in all_topics:
        topic_digits=re.sub(r'\D','',topic_id)
        rouge_output_filename=output_filename_base+topic_id+".txt"
        print ("Running rouge on specific topic: "+topic_id+" output: "+str(rouge_output_filename))

        r.system_dir = system_dir
        r.model_dir = TEMP_DATA_PATH+"2005/results/rouge/models" #(/duc/2005/results/ROUGE/models)
        r.system_filename_pattern = 'd('+topic_digits+')..txt'
        r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D

        output = r.convert_and_evaluate()

        fp=open(rouge_output_filename,'w')
        fp.write(output)
        fp.close()
        print(output)
        output_dict = r.output_to_dict(output)

    return
#==========================================================================================

def run_on_all_topics(system_dir=''):
    print ("[debug run rouge system dir]: "+str(system_dir))
    all_topic_output_filename=OUTPUT_ROUGE_DIR+"/rouge_all_topics.txt"

    #                     /data/
    r.system_dir = system_dir
    r.model_dir = TEMP_DATA_PATH+"2005/results/rouge/models" #(/duc/2005/results/ROUGE/models)
    r.system_filename_pattern = 'd(\d+)..txt'
    r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D
    
    print ("Using user trained summaries from: "+str(r.system_dir))
    print ("Using gold system trained summaries from: "+str(r.model_dir))
    
    if not os.path.exists(r.system_dir): print ("Bad system dir: "+r.system_dir)
    if not os.path.exists(r.model_dir): print ("Bad model dir: "+r.model_dir)
    
    output = r.convert_and_evaluate()

    fp=open(all_topic_output_filename,'w')
    fp.write(output)
    fp.close()
    print(output)
    output_dict = r.output_to_dict(output)
    print("Done evaluation")
    return

if __name__=='__main__':
    branches=['run_on_different_setups']
    for b in branches:
        globals()[b]()

print ("Done rouge")


"""
INSTALL NOTES:
    https://stackoverflow.com/questions/47045436/how-to-install-the-python-package-pyrouge-on-microsoft-windows

    strawberry perl
    http://strawberryperl.com/

    rebuild db
    You should remove \RELEASE-1.5.5\data\WordNet-2.0.exc.db, then from cmd.exe:
    cd RELEASE-1.5.5\data\
perl WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db

"""
