import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from dataset import KoBARTSummaryDataset
from torch.utils.data import DataLoader, Dataset
import textwrap
import string
import re
import pandas as pd

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./nl2url_v2.0.0')
    return model

model = load_model()
model.cuda()
tokenizer = get_kobart_tokenizer()

test_file_path = 'data/nl2url_v2.0.0_test.tsv'
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
    '''
    def lower(text):
        return text.lower()
    '''
    def rid_of_specials(text):
        text = text.replace("<extra_id_0>", "")
        text = text.replace("<extra_id_1>", "")
        return text

    return rid_of_specials(white_space_fix(remove_articles(remove_punc(s))))

def post_processing(text):
    if text.find('지역특화') >= 0:
        text = re.sub('항 공 [0-9]+ 김포(공)', '항 공 0110 김포(공)', text)
        text = re.sub('항 공 [0-9]+ 인천(공)', '항 공 0113 인천(공)', text)
        text = re.sub('항 공 [0-9]+ 울산(공)', '항 공 0151 울산(공)', text)
        text = re.sub('항 공 [0-9]+ 무안(공)', '항 공 0163 무안(공)', text)
        text = re.sub('항 공 [0-9]+ 여수(공)', '항 공 0167 여수(공)', text)
        text = re.sub('항 공 [0-9]+ 양양(공)', '항 공 0092 양양(공)', text)
        text = re.sub('항 공 [0-9]+ 제주(공)', '항 공 0182 제주(공)', text)
        text = re.sub('기상청 [0-9]+ 동두천', '기상청 0098 동두천', text)
        text = re.sub('기상청 [0-9]+ 서울', '기상청 0108 서울', text)
        text = re.sub('기상청 [0-9]+ KMA', '기상청 0410 KMA', text)
        text = re.sub('기상청 [0-9]+ 인천', '기상청 0112 인천', text)
        text = re.sub('기상청 [0-9]+ 수원', '기상청 0119 수원', text)
        text = re.sub('기상청 [0-9]+ 강화', '기상청 0201 강화', text)
        text = re.sub('기상청 [0-9]+ 양평', '기상청 0202 양평', text)
        text = re.sub('기상청 [0-9]+ 이천', '기상청 0203 이천', text)
        text = re.sub('기상청 [0-9]+ 안성', '기상청 0516 안성', text)
        text = re.sub('미공군 [0-9]+ 문산기상', '미공군 0380 문산기상', text)
        text = re.sub('미공군 [0-9]+ 광주초월', '미공군 0386 광주초월', text)
        text = re.sub('미공군 [0-9]+ 양평서종', '미공군 0395 양평서종', text)
        text = re.sub('미공군 [0-9]+ 양주광적', '미공군 0398 양주광적', text)
        text = re.sub('해군목측 [0-9]+ 평택', '해군목측 2101 평택', text)
        text = re.sub('해군목측 [0-9]+ 백령도', '해군목측 2102 백령도', text)
        text = re.sub('해군목측 [0-9]+ 연평도', '해군목측 2103 연평도', text)
        text = re.sub('해군목측 [0-9]+ 연평도남쪽', '해군목측 2201 연평도남쪽', text)
        text = re.sub('해군목측 [0-9]+ 덕적도', '해군목측 2104 덕적도', text)
        text = re.sub('해군목측 [0-9]+ 인천', '해군목측 2106 인천', text)
        text = re.sub('해군목측 [0-9]+ 소청도', '해군목측 2202 소청도', text)
        text = re.sub('해군목측 [0-9]+ 백령도(서)', '해군목측 2203 백령도(서)', text)
        text = re.sub('기상청 [0-9]+ 울릉도', '기상청 0115 울릉도', text)
        text = re.sub('기상청 [0-9]+ 안동', '기상청 0136 안동', text)
        text = re.sub('기상청 [0-9]+ 포항', '기상청 0138 포항', text)
        text = re.sub('기상청 [0-9]+ 대구', '기상청 0143 대구', text)
        text = re.sub('기상청 [0-9]+ 울산', '기상청 0152 울산', text)
        text = re.sub('기상청 [0-9]+ 창원', '기상청 0155 창원', text)
        text = re.sub('기상청 [0-9]+ 부산', '기상청 0159 부산', text)
        text = re.sub('기상청 [0-9]+ 통영', '기상청 0162 통영', text)
        text = re.sub('기상청 [0-9]+ 진주', '기상청 0192 진주', text)
        text = re.sub('기상청 [0-9]+ 봉화', '기상청 0271 봉화', text)
        text = re.sub('기상청 [0-9]+ 영주', '기상청 0272 영주', text)
        text = re.sub('기상청 [0-9]+ 문경', '기상청 0273 문경', text)
        text = re.sub('기상청 [0-9]+ 거제', '기상청 0294 거제', text)
        text = re.sub('기상청 [0-9]+ 남해', '기상청 0295 남해', text)
        text = re.sub('기상청 [0-9]+ 의성', '기상청 0278 의성', text)
        text = re.sub('기상청 [0-9]+ 구미', '기상청 0279 구미', text)
        text = re.sub('기상청 [0-9]+ 영천', '기상청 0281 영천', text)
        text = re.sub('기상청 [0-9]+ 거창', '기상청 0284 거창', text)
        text = re.sub('기상청 [0-9]+ 합천', '기상청 0285 합천', text)
        text = re.sub('기상청 [0-9]+ 밀양', '기상청 0288 밀양', text)
        text = re.sub('기상청 [0-9]+ 산청', '기상청 0289 산청', text)
        text = re.sub('미공군 [0-9]+ 청도화양', '미공군 0394 청도화양', text)
        text = re.sub('해군목측 [0-9]+ 죽변', '해군목측 1103 죽변', text)
        text = re.sub('해군목측 [0-9]+ 구룡포', '해군목측 1104 구룡포', text)
        text = re.sub('해군목측 [0-9]+ 울릉도', '해군목측 1105 울릉도', text)
        text = re.sub('해군목측 [0-9]+ 포항', '해군목측 1107 포항', text)
        text = re.sub('해군목측 [0-9]+ 독도', '해군목측 1204 독도', text)
        text = re.sub('해군목측 [0-9]+ 부산', '해군목측 3101 부산', text)
        text = re.sub('해군목측 [0-9]+ 욕지도', '해군목측 3103 욕지도', text)
        text = re.sub('해군목측 [0-9]+ 영도', '해군목측 3109 영도', text)
        text = re.sub('해군목측 [0-9]+ 욕지도부근', '해군목측 3202 욕지도부근', text)
        text = re.sub('기상청 [0-9]+ 군산', '기상청 0140 군산', text)
        text = re.sub('기상청 [0-9]+ 전주', '기상청 0146 전주', text)
        text = re.sub('기상청 [0-9]+ 광주', '기상청 0156 광주', text)
        text = re.sub('기상청 [0-9]+ 목포', '기상청 0165 목포', text)
        text = re.sub('기상청 [0-9]+ 여수', '기상청 0168 여수', text)
        text = re.sub('기상청 [0-9]+ 완도', '기상청 0170 완도', text)
        text = re.sub('기상청 [0-9]+ 진도', '기상청 0175 진도', text)
        text = re.sub('기상청 [0-9]+ 부안', '기상청 0243 부안', text)
        text = re.sub('기상청 [0-9]+ 임실', '기상청 0244 임실', text)
        text = re.sub('기상청 [0-9]+ 정읍', '기상청 0245 정읍', text)
        text = re.sub('기상청 [0-9]+ 남원', '기상청 0247 남원', text)
        text = re.sub('기상청 [0-9]+ 장수', '기상청 0248 장수', text)
        text = re.sub('기상청 [0-9]+ 순천', '기상청 0174 순천', text)
        text = re.sub('기상청 [0-9]+ 장흥', '기상청 0260 장흥', text)
        text = re.sub('기상청 [0-9]+ 해남', '기상청 0261 해남', text)
        text = re.sub('기상청 [0-9]+ 고흥', '기상청 0262 고흥', text)
        text = re.sub('해군목측 [0-9]+ 거문도', '해군목측 3104 거문도', text)
        text = re.sub('해군목측 [0-9]+ 흑산도', '해군목측 3106 흑산도', text)
        text = re.sub('해군목측 [0-9]+ 목포', '해군목측 3107 목포', text)
        text = re.sub('해군목측 [0-9]+ 흑산도부근', '해군목측 3204 흑산도부근', text)
        text = re.sub('기상청 [0-9]+ 충주', '기상청 0127 충주', text)
        text = re.sub('기상청 [0-9]+ 서산', '기상청 0129 서산', text)
        text = re.sub('기상청 [0-9]+ 청주', '기상청 0131 청주', text)
        text = re.sub('기상청 [0-9]+ 대전', '기상청 0133 대전', text)
        text = re.sub('기상청 [0-9]+ 추풍령', '기상청 0135 추풍령', text)
        text = re.sub('기상청 [0-9]+ 제천', '기상청 0221 제천', text)
        text = re.sub('기상청 [0-9]+ 보은', '기상청 0226 보은', text)
        text = re.sub('기상청 [0-9]+ 천안', '기상청 0232 천안', text)
        text = re.sub('기상청 [0-9]+ 부여', '기상청 0236 부여', text)
        text = re.sub('기상청 [0-9]+ 금산', '기상청 0238 금산', text)
        text = re.sub('기상청 [0-9]+ 청원', '기상청 0624 청원', text)
        text = re.sub('기상청 [0-9]+ 태안', '기상청 0627 태안', text)
        text = re.sub('미공군 [0-9]+ 금산군청', '미공군 0390 금산군청', text)
        text = re.sub('해군목측 [0-9]+ 어청도', '해군목측 2105 어청도', text)
        text = re.sub('해군목측 [0-9]+ 계룡대', '해군목측 4101 계룡대', text)
        text = re.sub('기상청 [0-9]+ 속초', '기상청 0090 속초', text)
        text = re.sub('기상청 [0-9]+ 철원', '기상청 0095 철원', text)
        text = re.sub('기상청 [0-9]+ 대관령', '기상청 0100 대관령', text)
        text = re.sub('기상청 [0-9]+ 춘천', '기상청 0101 춘천', text)
        text = re.sub('기상청 [0-9]+ 원주', '기상청 0114 원주', text)
        text = re.sub('기상청 [0-9]+ 영월', '기상청 0121 영월', text)
        text = re.sub('기상청 [0-9]+ 태백', '기상청 0216 태백', text)
        text = re.sub('미공군 [0-9]+ 김화학사', '미공군 0381 김화학사', text)
        text = re.sub('미공군 [0-9]+ 양구남면', '미공군 0382 양구남면', text)
        text = re.sub('미공군 [0-9]+ 인제신남', '미공군 0384 인제신남', text)
        text = re.sub('미공군 [0-9]+ 홍천기상', '미공군 0397 홍천기상', text)
        text = re.sub('해군목측 [0-9]+ 동해', '해군목측 1101 동해', text)
        text = re.sub('해군목측 [0-9]+ 기사문', '해군목측 1102 기사문', text)
        text = re.sub('해군목측 [0-9]+ 속초', '해군목측 1201 속초', text)
        text = re.sub('해군목측 [0-9]+ 주문진', '해군목측 1202 주문진', text)
        text = re.sub('기상청 [0-9]+ 제주', '기상청 0184 제주', text)
        text = re.sub('기상청 [0-9]+ 성산', '기상청 0188 성산', text)
        text = re.sub('기상청 [0-9]+ 서귀포', '기상청 0189 서귀포', text)
        text = re.sub('해군목측 [0-9]+ 추자도', '해군목측 3105 추자도', text)
        text = re.sub('해군목측 [0-9]+ 제주', '해군목측 3108 제주', text)
        text = re.sub('해군목측 [0-9]+ 서귀포', '해군목측 3110 서귀포', text)
        text = re.sub('해군목측 [0-9]+ 제주남부', '해군목측 3203 제주남부', text)
    return text


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
results = []

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
        predicted = post_processing(predicted)
        em = exact_match_score(predicted, ground_truth)
        subset = approx_match_score(predicted, ground_truth)         
        print(f'{total_cnt} INPUT : {lines}')
        print(f'GROUD TRUTH: {ground_truth}, MODEL OUTPUT: {predicted}')
        if em == 1:
            em_correct_num+=1
        else:
            result = [lines, ground_truth, predicted]
            results.append(result)
        if subset == 1:
            subset_correct_num+=1
df = pd.DataFrame(results, columns=['input', 'ground_truth', 'predicted'])

# df.to_excel('data/results_wrong.xlsx', index=True)

print(f'Number of total validation data: {total_cnt}')
print(f'Number of correct predictions: {em_correct_num, subset_correct_num}. Percentage : {em_correct_num / total_cnt, subset_correct_num / total_cnt}')