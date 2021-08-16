import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from dataset import KoBARTSummaryDataset
from torch.utils.data import DataLoader, Dataset
import textwrap
import string
import re

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    return model

model = load_model()
model.cuda()
tokenizer = get_kobart_tokenizer()

test_file_path = 'data/weather_test.tsv'
test = KoBARTSummaryDataset(test_file_path, tokenizer, 500)
val_loader = DataLoader(test, batch_size=8, num_workers=4, shuffle=False)
pad_token_id = 0

def clean_up(text):
    '''
    
    text = text.replace(".", '')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    '''
    text = text.replace("<s>", "")
    text =text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = text.replace('<usr>', '')
    return text  

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def rid_of_specials(text):
        text = text.replace("<extra_id_0>", "")
        text = text.replace("<extra_id_1>", "")
        return text

    return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def approx_match_score(prediction, ground_truth):
    answer = normalize_answer(prediction) 
    gt = normalize_answer(ground_truth)
    match = 0
    gt_words = gt.split(" ")
    for word in gt_words:
        if word in answer:
            match = 1
            return match
    return match 

total_cnt=0
em_correct_num = 0
subset_correct_num = 0

for batch in iter(val_loader):
    attention_mask = batch['input_ids'].ne(pad_token_id).float()
    decoder_attention_mask = batch['decoder_input_ids'].ne(pad_token_id).float()

    outs = model.generate(
        batch["input_ids"].cuda(),
        attention_mask=attention_mask.cuda(),
        use_cache=True,
        decoder_attention_mask=decoder_attention_mask.cuda(),
        max_length=100,
        num_beams=20,
        #early_stopping=True,
        #no_repeat_ngram_size=3
    )
    target2 = []
    for ids in batch['labels']:
        new_ids = [0 if x == -100 else x for x in ids]
        target2.append(new_ids)

    dec = [tokenizer.decode(ids) for ids in outs]
    texts = [tokenizer.decode(ids) for ids in batch['input_ids']]
    #targets = [tokenizer.decode(ids, for ids in batch['labels']]
    targets = [tokenizer.decode(ids) for ids in target2]

    for i in range(len(batch['input_ids'])):
        total_cnt+=1
        lines = textwrap.wrap("\n%s\n" % texts[i], width=3000)
        lines = clean_up(lines[0])
        ground_truth = clean_up(targets[i])
        predicted = clean_up(dec[i])
        em = exact_match_score(predicted, ground_truth)
        subset = approx_match_score(predicted, ground_truth)         
        print(f'{total_cnt} INPUT : {lines}')
        print(f'GROUD TRUTH: {ground_truth}, MODEL OUTPUT: {predicted}')
        if em == 1:
            em_correct_num+=1
        if subset == 1:
            subset_correct_num+=1

print(f'Number of total validation data: {total_cnt}')
print(f'Number of correct predictions: {em_correct_num, subset_correct_num}. Percentage : {em_correct_num / total_cnt, subset_correct_num / total_cnt}')