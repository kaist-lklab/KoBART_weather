import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util 
from tokenizers import Tokenizer
from grammar_regex import is_correct_grammar
import pandas as pd
from random import randrange
from pprint import pprint
import numpy as np

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_weather_v2')
    model2 = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model, model2

def get_tokenizer():
    return get_kobart_tokenizer()

def get_sql(input, templates):
    template_embeds = templates[0]
    index_to_input = templates[1]
    template_dict = templates[2]
    
    embeds = model2.encode(input)
    
    #Compute cosine-similarities for input and input templates for matching
    cosine_scores = util.pytorch_cos_sim(embeds, template_embeds)
    indx = np.argmax((cosine_scores.numpy())[0]) 
    input_template = index_to_input[indx]
    ot = template_dict[input_template] #matched sql template

    checkpoints = []
    j=0
    # Getting checkpoints between matched template & input
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
            if template_var in ot:
                ot = ot.replace(template_var, input_var)
    return ot
    
def get_output(input, templates):
    text = input['source']
    date_s = input['date'].split(" ")
    ymd = date_s[0].split('-')
    hms = date_s[1].split(':')
    date = [ymd[0],ymd[1],ymd[2],hms[0],hms[1]]
    # Get rid of date information input
    original_text = text
    if '-' in text and ':' in text:
        text = text[17:]
    elif '-' in text:
        text = text[11:]
    elif ':' in text:
        text = text[6:]
        
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids).to('cuda')
    input_ids = input_ids.unsqueeze(0)
    outputs = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5, num_return_sequences=5)
    res = []
    for output in outputs:
        res.append(tokenizer.decode(output, skip_special_tokens=True))
    out = None
    for x in res:
        if is_correct_grammar(x):
            criteria_met = True
            out = x
            break
    if out==None:
        out = res[0]
    sql = get_sql(text, templates)
    return [original_text, out, date, sql]

def response_template(res):
    input = res[0]
    output = res[1]
    date = res[2]
    sql = res[3]
    if date!=[]:
        year = date[0]
        month = date[1]
        day = date[2]
        hour = date[3]
        minute = date[4]
        if '내일' in input:
            day = str(int(day) + 1)
        if '어제' in input:
            day = str(int(day) - 1)
        #handling custom year-month-day time" 
        # There is date information included in input
        if '-' in input:
            indx = input.find('-')
            year = input[indx-4:indx]
            month = input[indx+1:indx+3]
            day = input[indx+4:indx+6]
        # There is time information included in input
        if ':' in input:
            indx = input.find(':')
            hour = input[indx-2:indx]
            minute = input[indx+1:indx+3]
        output = output.replace('YYYYMMDDHHMI', year+month+day+hour+minute)
    else:
        output = output.replace("입력='YYYYMMDDHHMI'", '')
    sql = sql.replace('YYYYMMDD', year+month+day)
    sql = sql.replace('MMDD', month+day)
    response = {
        "pseudoList":[{
            "site":"COMIS",
            "pseudo":output,
        }, {}],
        "extremeValue":[sql]
    }
    return response

def get_template_embeddings(model, data_dir):
    try:
        sql_template = pd.read_csv("data/template.csv")
    except:
        sql_template = pd.read_csv('home/KoBART-summarization/template.csv')
    template_dict = {}
    index_to_input = {}
    template_embeds = []

    # Getting templates
    for index,row in sql_template.iterrows(): 
        input = row['input'] 
        output = row['output'] 
        index_to_input[index] = input 
        template_dict[input] = output 
        template_embeds.append(model.encode(input)) 

    return (template_embeds, index_to_input, template_dict)



tokenizer = get_tokenizer()
if __name__ == '__main__':
    example = "2021-10-01 09:00 KIM전구 K Index"

    input = {
        "source" : example,
        "date" : "2021-11-16 00:00:00",
        "sourceType" : "text",
        "responseChannel": "aiw-response"
    }

    model, model2 = load_model()
    model = model.to('cuda')
    model2 = model2.to('cuda')

    tokenizer = get_tokenizer()
    templates = get_template_embeddings(model2, 'template.csv')

    output = response_template(get_output(input, templates))


    print('input:', input)
    print('output:', output)
