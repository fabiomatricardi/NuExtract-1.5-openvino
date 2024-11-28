"""
optimum-cli export openvino --model NuExtract-1.5-tiny --task text-generation-with-past --trust-remote-code --weight-format int8 ov_NuExtract-1.5-tiny

Followed official tutorial
https://docs.openvino.ai/2024/notebooks/llm-question-answering-with-output.html
"""
import streamlit as st
import warnings
warnings.filterwarnings(action='ignore')
import datetime
import random
import string
from time import sleep
import tiktoken
import json
import openvino_genai as ov_genai
# https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html
# https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html
# example from https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/chat_sample/chat_sample.py
# API documentation: https://docs.openvino.ai/2024/api/genai_api/_autosummary/openvino_genai.html#module-openvino_genai

# for counting the tokens in the prompt and in the result
encoding = tiktoken.get_encoding("cl100k_base") 
# GLOBALS
modelname = "NuExtract1.5-Tiny"
modelfile = 'ov_NuExtract-1.5-tiny'
# Set the webpage title
st.set_page_config(
    page_title=f"Your LocalGPT ✨ with {modelname}",
    page_icon="🌟",
    layout="wide")
# SET Session States
if "hf_model" not in st.session_state:
    st.session_state.hf_model = modelname
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repeat" not in st.session_state:
    st.session_state.repeat = 1.35
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1
if "maxlength" not in st.session_state:
    st.session_state.maxlength = 500
if "speed" not in st.session_state:
    st.session_state.speed = 0.0
if "time" not in st.session_state:
    st.session_state.time = ''

if "firstrun" not in st.session_state:
    st.session_state.firstrun = 0           
# Defining internal functions
def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()
def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res
# CACHED RESOURCES
@st.cache_resource 
def create_chat():   
# Set HF API token  and HF repo
    #device = 'CPU'  # GPU can be used as well
    #pipe = openvino_genai.LLMPipeline('ov_NuExtract-1.5-tiny', device)
    start = datetime.datetime.now()
    model_dir = 'ov_NuExtract-1.5-tiny'
    pipe = ov_genai.LLMPipeline(model_dir, 'CPU')
    delta = datetime.datetime.now() - start
    print(f'loading {modelfile} with pure Openvino-genAI pipeline in {delta}...')
    return pipe


# create the log file
if "logfilename" not in st.session_state:
    logfile = f'{genRANstring(5)}_log.txt'
    st.session_state.logfilename = logfile
    #Write in the history the first 2 sessions
    writehistory(st.session_state.logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT JSON extractor with 🌀 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
    writehistory(st.session_state.logfilename,f'🌀: How may I help you today?')

# INSTANTIATE THE API CLIENT to the LLM
llm = create_chat()

### START STREAMLIT UI
# Create a header element
mytitle = f'## Extract data with {modelname}'
st.markdown(mytitle, unsafe_allow_html=True)
# CREATE THE SIDEBAR
with st.sidebar:
    st.session_state.temperature = st.slider('Temperature:', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    st.session_state.maxlength = st.slider('Length reply:', min_value=150, max_value=2000, 
                                           value=500, step=50)
    st.session_state.presence = st.slider('Repeat Penalty:', min_value=0.0, max_value=2.0, value=1.11, step=0.02)
    st.markdown(f"**Logfile**: {st.session_state.logfilename}")
    statspeed = st.markdown(f'💫 speed: {st.session_state.speed}  t/s')
    gentime = st.markdown(f'⏱️ gen time: {st.session_state.time}  seconds')
    btnClear = st.button("Load example",type="primary", use_container_width=True)
    st.image('logo.png', use_container_width=True)

# MAIN WINDOWN
st.session_state.jsonformat = st.text_area('JSON Schema to be applied', value="", height=150,   
                     placeholder='here your schema', disabled=False, label_visibility="visible")
st.session_state.origintext = st.text_area('Source Document', value="", height=150, 
                     placeholder='here your text', disabled=False, label_visibility="visible")
extract_btn = st.button("Extract Data",type="primary", use_container_width=False)
st.markdown('---')
st.session_state.extractedJSON = st.empty()
st.session_state.onlyJSON = st.empty()

def create_example():
    jsontemplate = """{
    "Model": {
        "Name": "",
        "Number of parameters": "",
        "Number of max token": "",
        "Architecture": []
    },
    "Usage": {
        "Use case": [],
        "Licence": ""
    }
}"""
    text = """We introduce Mistral 7B, a 7–billion-parameter language model engineered for
superior performance and efficiency. Mistral 7B outperforms the best open 13B
model (Llama 2) across all evaluated benchmarks, and the best released 34B
model (Llama 1) in reasoning, mathematics, and code generation. Our model
leverages grouped-query attention (GQA) for faster inference, coupled with sliding
window attention (SWA) to effectively handle sequences of arbitrary length with a
reduced inference cost. We also provide a model fine-tuned to follow instructions,
Mistral 7B – Instruct, that surpasses Llama 2 13B – chat model both on human and
automated benchmarks. Our models are released under the Apache 2.0 license.
Code: <https://github.com/mistralai/mistral-src>
Webpage: <https://mistral.ai/news/announcing-mistral-7b/>"""
  
    st.session_state.jsonformat=jsontemplate
    st.session_state.origintext=text

# ACTIONS

#if btnClear:
#    create_example()

if extract_btn:
        prompt = f"""<|input|>\n### Template:
{st.session_state.jsonformat}

### Text:
{st.session_state.origintext}
<|output|>
"""
        print(prompt)
        with st.spinner("Thinking..."):
            start =  datetime.datetime.now()
            # https://platform.openai.com/docs/api-reference/completions/create
            output = llm.generate(prompt, temperature=st.session_state.temperature, 
                        do_sample=True, 
                        max_new_tokens=st.session_state.maxlength, 
                        repetition_penalty=st.session_state.presence,
                        eos_token_id = 151643)

        delta = datetime.datetime.now() -start
        print(output)
        result = output
        st.write(result)
        #adapter = result #.replace("'",'"')
        #final = json.loads(adapter) 
        totalTokens = len(encoding.encode(prompt))+len(encoding.encode(result))
        totalseconds = delta.total_seconds()
        st.session_state.time = totalseconds
        st.session_state.speed = totalTokens/totalseconds
        statspeed.markdown(f'💫 speed: {st.session_state.speed:.2f}  t/s')
        gentime.markdown(f'⏱️ gen time: {st.session_state.time:.2f}  seconds')
        totalstring = f"""GENERATED STRING

{result}
---

Generated in {delta}

---

JSON FORMAT:
"""   
        #WRITE THE OUTPUT AND THE LOGS
        st.session_state.onlyJSON.json(result)    
        writehistory(st.session_state.logfilename,f'✨: {prompt}')
        writehistory(st.session_state.logfilename,f'🌀: {result}')
        writehistory(st.session_state.logfilename,f'---\n\n')
