import re
import os
import codecs


from ex_all_topics import run_exercise
from rouge.do_rouge import run_on_all_topics
from duc_reader import TEMP_DATA_PATH

def get_fscore():
    filename_rouge_all_topics='./rouge/rouge_results/rouge_all_topics.txt'
    fp=open(filename_rouge_all_topics)

    avgf=0
    for line in fp.readlines():
        line=line.strip()
        
        m=re.search(r'1 Average_F. (.*) \(',line)
        if m:
            avgf=m.group(1)
            print ("AVGF: "+str(avgf))
            #1 ROUGE-1 Average_R: 0.37033 (95%-conf.int. 0.35567 - 0.38477)
            #print ("> "+str(line))
    fp.close()
    return avgf


def rouge_analysis(topic_ids):
    avgf=get_fscore()
    return avgf


def run_grid_search():
    branches=['pipeline']
    branches=['rouge']
    branches=['analysis']

    topic_id='d354c'
    
    if 'pipeline' in branches:
        branch_removal=['create_sim_matrix']
        branch_removal=[]
        run_exercise(force_topic_id=topic_id,branch_removal=branch_removal)
    
    if 'rouge' in branches:
        ## Run rouge for specific topic...
        #run_on_different_setups()
        topic_digits=re.sub(r'\D',r'',str(topic_id))
        print ("Running rouge for topic digits: "+str(topic_digits))
        system_dir = TEMP_DATA_PATH+"Top_summary"
        local_system_dir=system_dir+"/ex_do6_two_scores_1/leading_eigenvector"

        run_on_all_topics(system_dir=local_system_dir,topic_regex=topic_digits)
        print ("**WATCH THAT SUMMARIES ACTUALLY IN: "+str(local_system_dir))
        
    if 'analysis' in branches:
        avgf=rouge_analysis([topic_id])
    
    print ("Done grid search")
    return


if __name__=='__main__':
    branches=['run_grid_search']
    for b in branches:
        globals()[b]()
        
        
        
        
























