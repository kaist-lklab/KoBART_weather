import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from tokenizers import Tokenizer


def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_weather_v2')
    return model

def get_tokenizer():
    return get_kobart_tokenizer()

def get_output(input):
    text = input['source']
    date_s = input['date'].split(" ")
    ymd = date_s[0].split('-')
    hms = date_s[1].split(':')
    date = [ymd[0],ymd[1],ymd[2],hms[0],hms[1]]
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids).to('cuda')
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5, num_return_sequences=1)
    res = tokenizer.decode(output[0], skip_special_tokens=True)
    return [text, res, date]

def response_template(res):
    input = res[0]
    output = res[1]
    date = res[2]
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
        output = output.replace('YYYYMMDDHHMI', year+month+day+hour+minute)
    else:
        output = output.replace("입력='YYYYMMDDHHMI'", '')
    response = {
        "pseudoList":[{
            "site":"COMIS",
            "pseudo":output,
        }, {}],
        "extremeValue":[]
    }
    return response

input = {
    "source" : "오늘의 UM전구 저기압이동경로",
    "date" : "2021-10-18 00:00:00",
    "sourceType" : "text",
    "responseChannel": "aiw-response"
}

model = load_model()
model = model.to('cuda')
tokenizer = get_tokenizer()

output = response_template(get_output(input))

print('input:', input)
print('output:', output)