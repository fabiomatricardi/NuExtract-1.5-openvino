<img src='https://github.com/fabiomatricardi/NuExtract-1.5-openvino/blob/main/logo.png' width=400>

# NuExtract-1.5-openvino
Extract data with LLM and openvino

NuExtract-tiny-v1.5 is a fine-tuning of Qwen/Qwen2.5-0.5B, trained on a private high-quality dataset for structured information extraction. It supports long documents and several languages (English, French, Spanish, German, Portuguese, and Italian). To use the model, provide an input text and a JSON template describing the information you need to extract.

Note: This model is trained to prioritize pure extraction, so in most cases all text generated by the model is present as is in the original text.


### Convert the model Openvino IR
Install dependencies in a new virtual environment
```
# Step 1: Create virtual environment
python -m venv venv
# Step 2: Activate virtual environment
venv\Scripts\activate
# Step 3: Upgrade pip to latest version
python -m pip install --upgrade pip
# Step 4: Download and install the package
pip install openvino-genai==2024.3.0
```
This will install the dependencies for the runtime. But if you also want to prepare and optimize your model to OpenVino IR format you need also:
```
pip install optimum-intel[openvino] 
```
>Note that with the installation of the optimum-intel libraries, a bunch of other packages are included:
```
Installing collected packages: sentencepiece, pytz, ninja, mpmath, jstyleson, 
grapheme, xxhash, wrapt, watchdog, urllib3, tzdata, typing-extensions, 
tornado, toml, threadpoolctl, tenacity, tabulate, sympy, smmap, six, 
setuptools, safetensors, rpds-py, regex, pyyaml, pyreadline3, pyparsing, 
pygments, psutil, protobuf, pillow, numpy, networkx, natsort, narwhals, 
multidict, mdurl, MarkupSafe, kiwisolver, joblib, idna, fsspec, frozenlist, 
fonttools, filelock, dill, cycler, colorama, charset-normalizer, certifi, 
cachetools, blinker, attrs, aiohappyeyeballs, about-time, yarl, tqdm, scipy, 
requests, referencing, python-dateutil, pydot, pyarrow, onnx, multiprocess, 
markdown-it-py, jinja2, humanfriendly, gitdb, Deprecated, contourpy, cma, 
click, autograd, alive-progress, aiosignal, torch, tiktoken, scikit-learn, 
rich, pydeck, pandas, matplotlib, jsonschema-specifications, 
huggingface-hub, gitpython, coloredlogs, aiohttp, tokenizers, pymoo, 
jsonschema, transformers, nncf, datasets, altair, streamlit, optimum, 
optimum-intel
```

For the graphic interface also:
```
pip install tiktoken streamlit
```

## How to convert the model
In this repo I will explain how to convert the latest NuMind model called **NuExtract-1.5-tiny**

Download the model files from the official Hugging Face repository
[NuExtract-1.5-tiny
](https://huggingface.co/numind/NuExtract-1.5-tiny)

I put all the files into a subfolder called `NuExtract-1.5-tiny`

Open your terminal, and with the venv activated run:
```
optimum-cli export openvino --model NuExtract-1.5-tiny --task text-generation-with-past --trust-remote-code --weight-format int8 ov_NuExtract-1.5-tiny
```
This will create a new folder called `ov_NuExtract-1.5-tiny` with the IR model in quantized format `int8` together with its tokenizers

#### Resources
- [tutorial](https://docs.openvino.ai/2024/notebooks/llm-question-answering-with-output.html)
- [model-optimization-guide](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html) 
- [genai-guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) 
- [examples](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/chat_sample/chat_sample.py)
- [API documentation: ](https://docs.openvino.ai/2024/api/genai_api/_autosummary/openvino_genai.html#module-openvino_genai)


