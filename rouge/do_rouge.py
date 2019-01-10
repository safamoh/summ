import re
import os
import sys
from pyrouge import Rouge155

sys.path.insert(0,"..")
from duc_reader import get_list_of_all_topics
from duc_reader import TEMP_DATA_PATH
from duc_reader import DOCS_SOURCE


LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))+"/"

is_windows = sys.platform.startswith('win')

if is_windows:
    OUTPUT_ROUGE_DIR=LOCAL_PATH+"/rouge_results"
    r = Rouge155()
else:
    OUTPUT_ROUGE_DIR="/Users/safaalballaa/Desktop/resulted_files/rouge_results"
    r = Rouge155("/Users/safaalballaa/Desktop/test/summ/rouge/ROUGE-1.5.5")

print ("Rouge install ok!")

#My summaries:    system
#Gold summaries:  model


#PATTERN SETUP
#2005
#2006 Sample first file: C:/scripts-18/duc/summ/../data/2006/DUC2006_Summarization_Documents/duc2006_docs/D0601A/APW19990707.0181
#- topic: D\d\d\d\d.
#- file: A-Z....\.\d\d\d\d


def run_on_different_setups():

        
    system_dir = TEMP_DATA_PATH+"Top_summary"
    print ("Getting list of topics...")
    all_topics=get_list_of_all_topics()
    

    #branches=['do_random_walk']
    #branches=['select_top_cos_sims']
    #branches=['select_by_cluster_weight_factor']
    #branches=['do_selection_multiple_cluster_algs'] #top 3 sentence from each cluster
    #branches=['do_selection_by_round_robin']
    #branches=['do1_select_query_cluster']
    #branches=['do2_local_walk']
    #branches=['do3_avg_cosims']
    #branches=['do4_median_weight']
    #branches=['do5_markov_clustering']
#    branches=['do6_two_scores']
    branches=['do6_two_scores_1']
    #branches=['do6_two_scores_2']
    #branches=['do7_sum_nodes']
    


    #option='run_individual_topics'
    option='run_all_topics_together'



#====================================== BASELINE ====================================================
    if 'select_top_cos_sims' in branches:
        local_system_dir=system_dir+"/cos_sim"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)
#======================================== BASELINE ==================================================

    if 'do_random_walk' in branches:
        local_system_dir=system_dir+"/random_walk"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)


#==========================================================================================
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

    if 'do_selection_by_round_robin' in branches:
        local_system_dir=system_dir+"/round_robin/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)

#==========================================================================================
    if 'do1_select_query_cluster' in branches:
        local_system_dir=system_dir+"/ex_do1_select_query_cluster/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)
#==========================================================================================

    if 'do2_local_walk' in branches:
        local_system_dir=system_dir+"/ex_do2_local_walk/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)
#==========================================================================================

    if 'do3_avg_cosims' in branches:
        local_system_dir=system_dir+"/ex_do3_avg_cosims/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)

#==========================================================================================
    if 'do4_median_weight' in branches:
        local_system_dir=system_dir+"/ex_do4_median_weight/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)
#==========================================================================================

    if 'do5_markov_clustering' in branches:
        local_system_dir=system_dir+"/ex_do5_markov_clustering/markov"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)

#==========================================================================================

    if 'do6_two_scores' in branches:
        local_system_dir=system_dir+"/ex_do6_two_scores/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)

#==========================================================================================

    if 'do6_two_scores_1' in branches:
        local_system_dir=system_dir+"/ex_do6_two_scores_1/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)

#==========================================================================================

    if 'do6_two_scores_2' in branches:
        local_system_dir=system_dir+"/ex_do6_two_scores_2/leading_eigenvector"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)

#==========================================================================================

    if 'do7_sum_nodes' in branches:
        local_system_dir=system_dir+"/ex_do7_sum_nodes"
        output_filename_base=OUTPUT_ROUGE_DIR

        if 'run_individual_topics'==option:
            run_one_topic_at_a_time(system_dir=local_system_dir,output_filename_base=output_filename_base,all_topics=all_topics)
        else:
            print ("Running rouge for all topics together")
            run_on_all_topics(system_dir=local_system_dir)

    return

#==========================================================================================

def run_one_topic_at_a_time(system_dir='',output_filename_base='',all_topics=[]):
    upgrade_as=needed_stop
    for topic_id in all_topics:
        rouge_output_filename=output_filename_base+topic_id+".txt"
        print ("Running rouge on specific topic: "+topic_id+" output: "+str(rouge_output_filename))

        r.system_dir = system_dir
        r.model_dir = TEMP_DATA_PATH+DOCS_SOURCE+"/results/rouge/models" #(/duc/2005/results/ROUGE/models)
        if not os.path.exists(r.model_dir):
            os.makedirs(r.model_dir)
    
        if DOCS_SOURCE=='2005':
            topic_digits=re.sub(r'\D','',topic_id)
            r.system_filename_pattern = 'd('+topic_digits+')..txt'
            r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D
        elif DOCS_SOURCE=='2006':
            validate_search=hard_stop
            topic_digits=re.sub(r'\D','',topic_id)
            r.system_filename_pattern = 'd('+topic_digits+')..txt'
            r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D
        elif DOCS_SOURCE=='2007':
            validate_search=hard_stop
            topic_digits=re.sub(r'\D','',topic_id)
            r.system_filename_pattern = 'd('+topic_digits+')..txt'
            r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D
        else: requires_setup=hard_stop


        output = r.convert_and_evaluate()
        print ("OUTPUT: "+str(output))

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

    r.system_dir = system_dir
    if DOCS_SOURCE=='2005':
        the_dir=TEMP_DATA_PATH+DOCS_SOURCE+"/results/rouge/models" #(/duc/2005/results/ROUGE/models)
    elif DOCS_SOURCE=='2006':
        the_dir=TEMP_DATA_PATH+"../"+DOCS_SOURCE+"/nisteval/rouge/peers" #(/duc/2005/results/ROUGE/models)
    elif DOCS_SOURCE=='2007':
        the_dir=TEMP_DATA_PATH+"../"+DOCS_SOURCE+"/maineval/rouge/peers" #(/duc/2005/results/ROUGE/models)

    if not os.path.exists(the_dir):
        print ("Please update this path to point to gold-standard rouge models: "+str(the_dir))

    print ("gold standard models: "+str(the_dir))
    r.model_dir = the_dir

    if DOCS_SOURCE=='2005':
        r.system_filename_pattern = 'd(\d+)..txt'
        r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D
    elif DOCS_SOURCE=='2006':
        r.system_filename_pattern = 'D(\d+)..txt'
        r.model_filename_pattern = 'D#ID#.*[A-Z]' #D311.M.250.I.D
    elif DOCS_SOURCE=='2007':
        r.system_filename_pattern = 'D(\d+)..txt'
        r.model_filename_pattern = 'D#ID#.*[A-Z]'
    else: requires_setup=hard_stop
    
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
    cd RELEASE-1.5.5\data
perl WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db

"""
