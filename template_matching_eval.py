from sentence_transformers import SentenceTransformer, util 
import pandas as pd 
import numpy as np 

template = pd.read_csv("data/template.csv")
evaluation = pd.read_csv("data/evaluation.csv")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

template_dict = {}
index_to_input = {}
template_embeds = []

# Getting templates
for index,row in template.iterrows(): 
    input = row['input'] 
    output = row['output'] 
    index_to_input[index] = input 
    template_dict[input] = output 
    template_embeds.append(model.encode(input)) 

#Getting evaluation data
tm_correct = 0
sf_correct = 0

for index,row in evaluation.iterrows(): 
    template_id = (row['template'] - 1)
    input = row['input'] 
    output = row['output'] 
    embeds = model.encode(input)
    
    #Compute cosine-similarities for input and input templates for matching
    cosine_scores = util.pytorch_cos_sim(embeds, template_embeds)
    indx = np.argmax((cosine_scores.numpy())[0]) 
    input_template = index_to_input[indx]
    output_template = template_dict[input_template]
    if indx == template_id:
        ot = output_template
        tm_correct+=1
        print(input_template)
        print(input)

        checkpoints = []
        j=0
        for i in range(len(input_template)):
            if input_template[i] in input[j:]:
                j_ = input[j:].index(input_template[i])
                j = j + j_
                checkpoints.append([i,j])
                
        #Iterate through the checkpoints
        for i in range(len(checkpoints)-1):
            t1 = checkpoints[i][0]
            o1 = checkpoints[i][1]
            t2 = checkpoints[i+1][0]
            o2 = checkpoints[i+1][1]
            
            if (t1+1)!=t2:
                template_var = input_template[t1+1:t2]
                input_var = input[o1+1:o2]
                print(template_var, input_var)
                ot = ot.replace(template_var, input_var)
        
print(f'Percentage of correct template match :{tm_correct, len(evaluation)}')

