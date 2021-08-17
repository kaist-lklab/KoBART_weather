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
def get_output(text):
    global model
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids).to('cuda')
    input_ids = input_ids.unsqueeze(0)
    outputs = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5, num_return_sequences=5)
    res = []
    for output in outputs:
        res.append(tokenizer.decode(output, skip_special_tokens=True))
    return res

global model
model = load_model()
model = model.to('cuda')
tokenizer = get_tokenizer()

st.title("자연어-> COMIS 경로 매핑기")
#text_options = [
#        '함흥 (함경남도) UM국지의 단열선도',
#        'UM전구의 기압골, 기압능, 온도골, 온도능',
#        '제주 KIM전구의 단열선도',
#        '무안공항 UM전구와 ECM(WF)전구의 단열선도 비교',
#        '파주 KIM전구의 연직시계열(단기)',
#        '상주 ECM(WF)전구의 연직시계열(단기)',
#        '철원 ECM(WF)전구의 연직시계열중기',
#        ]

#c1, c2 = st.columns([9, 1])
#select_text = c1.selectbox("자연어 질문 예시", options=text_options)


#if select_text:
#    default_text = select_text
#else:
#    default_text = ''

#text = c2.text_input("자연어 질문 입력:(예: 제주 KIM전구의 단열선도)", value=default_text)
text = st.text_input("자연어 질문 입력: (예: 제주 KIM전구의 단열선도)")

if st.button('제출') or text:
    if not text:
        st.markdown("결과를 얻기 위해서는 자연어 원문을 써주세요.")
    else:
        st.markdown("## 자연어 원문")
        st.write(text)
        #text = text.replace('\n', '')
        st.markdown("## 경로 변환 결과 (top-5) ")
        #with st.spinner('processing..'):
        outputs = get_output(text)
        for i, output in enumerate(outputs):
            st.write(f"{i + 1}: {output}")
