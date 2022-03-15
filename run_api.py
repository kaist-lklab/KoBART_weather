import torch
from kobart import get_kobart_tokenizer
from flask import Flask, request, jsonify
from transformers.models.bart import BartForConditionalGeneration
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer, util
from grammar_regex import is_correct_grammar
import sys
import pandas as pd
import numpy as np
from pprint import pprint
import re
import os, psutil

process = psutil.Process(os.getpid())

print(process.memory_info().rss)
# from infer_dev import get_sql, get_output, get_template_embedding/, response_template
app = Flask(__name__)


def load_model():
    model = BartForConditionalGeneration.from_pretrained('home/KoBART-summarization/kobart_weather_v2')
    model2 = SentenceTransformer('home/KoBART-summarization/sentence-model')
    return model, model2

def get_tokenizer():
    return get_kobart_tokenizer()

@app.route('/api/search/text', methods=['GET'])
def process_request():
    global model, templates
    global tokenizer
    text_input = {}
    text_input['source'] = request.args.get('source')
    text_input['date'] = request.args.get('date')
    if text_input is None:
        output = response_template("Input is not a JSON form")
    output = response_template(get_output(text_input, templates))
    return output

'''
example = "당일(2021년 1월 2일) 전지점 일단위 최고온도 30개"

input = {
    "source" : example,
    "date" : "2021-10-18 00:00:00",
    "sourceType" : "text",
    "responseChannel": "aiw-response"
}
'''

def get_template_embeddings(model):
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
    response = {
        "pseudoList":[{
            "site":"COMIS",                                 
            "pseudo":output,
        }, {}],
        "extremeValue":sql
    }                       
    return response

def get_output(input, templates):
    global tokenizer, model
    original_text = input['source']
    if re.search('[0-9]+년 [0-9]+월 [0-9]+일', original_text):
        source_date = re.findall('[0-9]+년 [0-9]+월 [0-9]+일', original_text)[0]
        source_year = source_date.split('년 ')[0]
        source_month = source_date.split('년 ')[1].split('월 ')[0]
        source_day = source_date.split('년 ')[1].split('월 ')[1].split('일')[0]
        if int(source_month) < 10:
            source_month = "0" + source_month
        if int(source_day) < 10:
            source_day = "0" + source_day
        date = [source_year, source_month, source_day, '00', '00']
    else:
        if input['date']!=None and input['date']!='':
            date_s = input['date'].split(" ")        
            ymd = date_s[0].split('-')                       
            hms = date_s[1].split(':')
            date = [ymd[0],ymd[1],ymd[2],hms[0],hms[1]]
        else:
            print('Error: Fill the date variable.')
    # Get rid of date information input
    text = original_text                
    '''
    if '-' in text and ':' in text:   
        text = text[17:]
    elif '-' in text:
        text = text[11:]    
    elif ':' in text:   
        text = text[6:]
    '''
    input_ids = tokenizer.encode(text)
    if use_cuda:
        input_ids = torch.tensor(input_ids).to('cuda')
    else:
        input_ids = torch.tensor(input_ids)
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


def get_sql(input, templates):
    global model2
    template_embeds = templates[0]       
    index_to_input = templates[1]    
    template_dict = templates[2]         
    embeds = model2.encode(input)     
    #Compute cosine-similarities for input and input templates for matching
    cosine_scores = util.pytorch_cos_sim(embeds, template_embeds)                                           
    indx = np.argmax((cosine_scores.numpy())[0])               
    input_template = index_to_input[indx]
    output_template = template_dict[input_template] #matched sql template          
    checkpoints = []
    j=0
    # Getting checkpoints between matched template & input
    for i in range(len(input_template)):     
        if input_template[i] in input[j:]:
            j_ = input[j:].index(input_template[i])
            j = j + j_
            checkpoints.append([i,j])
    ot = output_template
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
                if 'date' in template_var: #handling date
                    year_ = input_var.split('년')
                    year = year_[0]
                    month_ = year_[1].split('월') 
                    month = month_[0][1:]
                    if len(month)==1:
                        month = f'0{month}'
                    day_ = month_[1].split('일')
                    day = day_[0][1:]
                    if len(day)==1:
                        day = f'0{day}'
                    input_var = f"'{year+month+day}'"
                    ot = ot.replace(template_var, input_var)
                elif 'month' in template_var:
                    month= (input_var.split('월'))[0]
                    input_var = f"'{month}'"
                    ot = ot.replace(template_var, input_var)
                elif 'number' in template_var:
                    ot = ot.replace(template_var, input_var)
                else:
                    input_var = f"'{input_var}'"
                    ot = ot.replace(template_var, input_var)
    if ot == output_template:
        return []
    else:
        return [ot]

if __name__ == '__main__':
    global model, model2
    global tokenizer
    global use_cuda
    global templates
    argvs = sys.argv
    if len(argvs) != 3:
        raise Exception("You need to specify the port number and device info")
    portnum = int(sys.argv[1])
    device_info = str(sys.argv[2])
    if device_info == 'cpu':
        use_cuda = False
    elif device_info == 'gpu':
        use_cuda = True
    else:
        raise Exception(f"You need to choose between 'cpu' or 'gpu' for the device info, but got {use_cuda}")
    print("loading model..")
    model, model2 = load_model()
    print("loaded!")
    if use_cuda:
        model = model.to('cuda')
        model2 = model2.to('cuda')
    tokenizer = get_tokenizer()
    templates = get_template_embeddings(model2)
    app.run(host='0.0.0.0', port=portnum, debug=False)
'''
output = response_template(get_output(input))

print('input:', input)
print('output:', output)
'''
