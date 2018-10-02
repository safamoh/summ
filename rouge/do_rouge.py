import re
import os
import sys
from pyrouge import Rouge155

sys.path.insert(0,"..")
from duc_reader import get_list_of_all_topics
from duc_reader import TEMP_DATA_PATH

r = Rouge155()

print ("Rouge install ok!")

LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))+"/"

OUTPUT_ROUGE_DIR=LOCAL_PATH+"/rouge_results"


#0v1#  JC  Oct 2, 2018  Initial setup

#My summaries:    system
#Gold summaries:  model

def run_one_topic_at_a_time():
    print ("Getting list of topics...")
    for topic_id in get_list_of_all_topics():
        topic_digits=re.sub(r'\D','',topic_id)
        rouge_output_filename=OUTPUT_ROUGE_DIR+"/rouge_"+topic_id+".txt"
        print ("Running rouge on specific topic: "+topic_id+" output: "+str(rouge_output_filename))

        r.system_dir = TEMP_DATA_PATH+"Top_summary"         #Can't have subdirectories!
        r.model_dir = TEMP_DATA_PATH+"2005/results/rouge/models" #(/duc/2005/results/ROUGE/models)
        r.system_filename_pattern = 'top_10_walk_scores_d('+topic_digits+')..txt' #text.(\d+).txt'
        r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D

        output = r.convert_and_evaluate()

        fp=open(rouge_output_filename,'w')
        fp.write(output)
        fp.close()
        print(output)
        output_dict = r.output_to_dict(output)

        break

    return

def run_on_all_topics():
    all_topic_output_filename=OUTPUT_ROUGE_DIR+"/rouge_all_topics.txt"

    #                     /data/
    r.system_dir = TEMP_DATA_PATH+"Top_summary"         #Can't have subdirectories!
    r.model_dir = TEMP_DATA_PATH+"2005/results/rouge/models" #(/duc/2005/results/ROUGE/models)
    r.system_filename_pattern = 'top_10_walk_scores_d(\d+)..txt' #text.(\d+).txt'
    r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D
    
    if not os.path.exists(r.system_dir): print ("Bad system dir: "+r.system_dir)
    if not os.path.exists(r.model_dir): print ("Bad model dir: "+r.model_dir)
    
    output = r.convert_and_evaluate()

    fp=open(all_topic_output_filename,'w')
    fp.write(output)
    fp.close()
    print(output)
    output_dict = r.output_to_dict(output)
    return

if __name__=='__main__':
    branches=['run_one_topic_at_a_time']
    branches=['run_on_all_topics']
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
