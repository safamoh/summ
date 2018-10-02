import os
from pyrouge import Rouge155
r = Rouge155()

print ("Rouge install ok!")

LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))+"/"

#My summaries:    system
#Gold summaries:  model

r.system_dir = LOCAL_PATH+"../../data/Top_summary"         #Can't have subdirectories!
r.model_dir = LOCAL_PATH+"../../data/2005/results/rouge/models" #(/duc/2005/results/ROUGE/models)
r.system_filename_pattern = 'top_10_walk_scores_d(\d+)i.txt' #text.(\d+).txt'
r.model_filename_pattern = 'D370.[A-Z].#ID#.txt' #'text.[A-Z].#ID#.txt'
r.model_filename_pattern = 'D#ID#.[A-Z]' #D311.M.250.I.D

if not os.path.exists(r.system_dir): print ("Bad system dir: "+r.system_dir)
if not os.path.exists(r.model_dir): print ("Bad model dir: "+r.model_dir)

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)

print ("Done rouge")
