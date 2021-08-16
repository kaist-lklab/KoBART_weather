import torch
import streamlit as st
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from tokenizers import Tokenizer

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()})
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_weather')
    return model

@st.cache(hash_funcs={Tokenizer: lambda tokenizer: tokenizer.__dir__()})
def get_tokenizer():
    return get_kobart_tokenizer()

@st.cache
def get_output(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

model = load_model()
tokenizer = get_tokenizer()

text_options = [
        '함흥 (함경남도) UM국지의 단열선도',
        'UM전구의 기압골, 기압능, 온도골, 온도능',
        '제주 KIM전구의 단열선도',
        '무안공항 UM전구와 ECM(WF)전구의 단열선도 비교',
        '파주 KIM전구의 연직시계열(단기)',
        '상주 ECM(WF)전구의 연직시계열(단기)',
        '철원 ECM(WF)전구의 연직시계열중기',
        ]

st.title("자연어-> COMIS 경로 매핑기")
c1, c2 = st.columns([3, 4])
select_text = c1.selectbox("자연어 질문 예시", options=text_options)


if select_text:
    default_text = select_text
else:
    default_text = ''

text = c2.text_input("자연어 질문 입력:", value=default_text)
st.markdown("## 자연어 원문")
st.write(text)

if text != '':
    text = text.replace('\n', '')
    st.markdown("## 경로 변환 결과")
    #with st.spinner('processing..'):
    output = get_output(text)

    st.write(output)
