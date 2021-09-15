import pandas as pd
import torch
import streamlit as st
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from tokenizers import Tokenizer

@st.cache(allow_output_mutation=True,
        hash_funcs={torch.Tensor: lambda tensor: tensor.detach().cpu().numpy(),
    torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach().cpu().numpy()})
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_weather')
    return model

@st.cache(hash_funcs={Tokenizer: lambda tokenizer: tokenizer.__dir__()})
def get_tokenizer():
    return get_kobart_tokenizer()

@st.cache
def get_output(text, num_sequences=5):
    global model
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids).to('cuda')
    input_ids = input_ids.unsqueeze(0)
    outputs = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=10, num_return_sequences=num_sequences)
    res = []
    for output in outputs:
        res.append(tokenizer.decode(output, skip_special_tokens=True))
    return res

global model
model = load_model()
model = model.to('cuda')
model.eval()
tokenizer = get_tokenizer()

st.title("자연어-> COMIS 경로 매핑기")
text_options = [
        '함흥 (함경남도) UM국지의 단열선도',
        'UM전구의 기압골, 기압능, 온도골, 온도능',
        '제주 KIM전구의 단열선도',
        '무안공항 UM전구와 ECM(WF)전구의 단열선도 비교',
        '파주 KIM전구의 연직시계열(단기)',
        '상주 ECM(WF)전구의 연직시계열(단기)',
        '철원 ECM(WF)전구의 연직시계열중기',
        ]

check_box = st.checkbox("예시 보기")
c1, c2 = st.columns([7, 2])
#select_text = c1.selectbox("자연어 질문 예시", options=text_options)


#if select_text:
#    default_text = select_text
#else:
#    default_text = ''

#text = c2.text_input("자연어 질문 입력:(예: 제주 KIM전구의 단열선도)", value=default_text)
#st.markdown("자연어 질문 입력")
#text = st.text_input("자연어 질문 입력: (예: 제주 KIM전구의 단열선도)")
if check_box:
    text = c1.selectbox("자연어 질문 예시", options=text_options)
else:
    text = c1.text_input('자연어 질문 입력')
k = c2.number_input('생성할 결과 개수', min_value=1, max_value=10, value=5, step=1)

if text: #c2.button('제출')
    if not text:
        st.markdown("결과를 얻기 위해서는 자연어 원문을 써주세요.")
    else:
        #st.markdown("## 자연어 원문")
        #st.write(text)
        #text = text.replace('\n', '')
        #st.markdown("## 경로 변환 결과 (top-5) ")
        #with st.spinner('processing..'):
        outputs = get_output(text, num_sequences=k)
        #df = pd.DataFrame(outputs, columns=['생성 결과'])
        df = pd.DataFrame({"Top-k": list(range(1, k+1)), "생성 결과": outputs})#.set_index("생성 순위")
        df = df.assign(hack='').set_index('hack')
        #df.columns.name = 'idx'
        #df.style.apply(lambda x: ['background: lightgreen' if x['idx'] == 0 else '' for i in x], axis=1)
        #df.style.highlight_min(axis=1)
        #for i, output in enumerate(outputs):
        #    st.write(f"{i + 1}: {output}")
        def colorize(x):
            if x['Top-k'] == 1:
                return ['background: yellow', 'background:yellow']
            else:
                return ['', '']
        st.table(df)
        #st.table(df.style.apply(lambda x: ['background: yellow' if x == 1 else '' for i in x], axis=1))
        #st.table(df.style.apply(colorize, axis=1))
       # st.table(df.style.highlight_quantile(axis=1, q_left=0, color='##ffd75'))
       # st.table(df.style.highlight_min(axis=1))
