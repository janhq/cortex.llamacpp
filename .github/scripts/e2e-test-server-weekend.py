import requests
import json
import subprocess
import os
import logging
import sys
import random
import platform
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    stop: list[str]
    system_prompt: str
    user_prompt: str
    ai_prompt: str
    hf_url: str


model_configs = []

# cortexso/llama3
model_configs.append(ModelConfig("llama3", ["<|end_of_text|>", "<|eot_id|>"],
                                 "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                                 "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
                                 "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                                 "https://huggingface.co/cortexso/llama3/resolve/main/model.gguf"))
# cortexso/llama3.1
model_configs.append(ModelConfig("llama3.1", ["<|end_of_text|>", "<|eot_id|>", "<|eom_id|>"],
                                 "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                                 "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
                                 "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                                 "https://huggingface.co/cortexso/llama3.1/resolve/main/model.gguf"))
# cortexso/gemma
model_configs.append(ModelConfig("gemma", ["<end_of_turn>", "<eos>"],
                                 "",
                                 "<start_of_turn>user\n",
                                 "<end_of_turn>\n<start_of_turn>model",
                                 "https://huggingface.co/cortexso/gemma/resolve/main/model.gguf"))
# cortexso/gemma2
model_configs.append(ModelConfig("gemma2", ["<end_of_turn>", "<eos>"],
                                 "",
                                 "<start_of_turn>user\n",
                                 "<end_of_turn>\n<start_of_turn>model",
                                 "https://huggingface.co/cortexso/gemma2/resolve/main/model.gguf"))
# cortexso/phi3
model_configs.append(ModelConfig("phi3", ["<|end|>"],
                                 "",
                                 "<|user|>\n",
                                 "<|end|>\n<|assistant|>\n",
                                 "https://huggingface.co/cortexso/phi3/resolve/main/model.gguf"))
# cortexso/mistral
model_configs.append(ModelConfig("mistral", ["</s>"],
                                 "<s>",
                                 " [INST] ",
                                 " [/INST]",
                     "https://huggingface.co/cortexso/mistral/resolve/main/model.gguf"))
# cortexso/openhermes-2.5
model_configs.append(ModelConfig("openhermes-2.5", ["</s>"],
                                 "<|im_start|>system\n",
                                 "<|im_end|>\n<|im_start|>user\n",
                                 "<|im_end|>\n<|im_start|>assistant\n",
                     "https://huggingface.co/cortexso/openhermes-2.5/resolve/main/model.gguf"))
# cortexso/tinyllama
model_configs.append(ModelConfig("tinyllama", ["</s>"],
                                 "<|system|>\n",
                                 "<|user|>\n",
                                 "<|assistant|>",
                                 "https://huggingface.co/cortexso/tinyllama/resolve/main/model.gguf"))
# cortexso/qwen2
model_configs.append(ModelConfig("qwen2", [],
                                 "<|im_start|>system\n",
                                 "<|im_end|>\n<|im_start|>user\n",
                                 "<|im_end|>\n<|im_start|>assistant",
                                 "https://huggingface.co/cortexso/qwen2/resolve/main/model.gguf"))



n = len(sys.argv)
print("Total arguments passed:", n)
if n != 2:
    print("The number of arguments should == 2")
    exit(1)

BINARY_PATH = sys.argv[1]
if platform.system == 'Windows':
    BINARY_PATH += '.exe'
LLM_MODEL = 'chat_model'
EMBED_MODEL = 'embedding_model'
LLM_FILE_TO_SAVE = './' + LLM_MODEL + '.gguf'
EMBED_FILE_TO_SAVE = './' + EMBED_MODEL + '.gguf'

CONST_CTX_SIZE = 1024
CONST_USER_ROLE = "user"
CONST_ASSISTANT_ROLE = "assistant"


logging.basicConfig(filename='./test.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

chat_data = []


def request_post(req_data, url, is_stream=False):
    """ 
    function supports for POST request 
    """
    try:
        r = requests.post(url, json=req_data, stream=is_stream, timeout=3600)
        r.raise_for_status()
        if is_stream:
            if r.encoding is None:
                r.encoding = 'utf-8'

            res = ""
            for line in r.iter_lines(decode_unicode=True):
                if line and "[DONE]" not in line:
                    data = json.loads(line[5:])
                    content = data['choices'][0]['delta']['content']
                    res += content
            logging.info('{\'assistant\': \'%s\'}', res)
            chat_data.append({
                "role": CONST_ASSISTANT_ROLE,
                "content": res
            })
            # Can be an error when model generates gabarge data
            res_len = len(res.split())
            if res_len >= CONST_CTX_SIZE - 50:
                logging.warning(
                    "Maybe generated gabarge data: %s", str(res_len))
                # return False
        else:
            res_json = r.json()
            logging.info(res_json)

        if r.status_code == 200:
            return True
        else:
            logging.warning('{\'status_code\': %s}', str(r.status_code))
            return False
    except requests.exceptions.HTTPError as error:
        logging.error(error)
        return False


def request_get(url):
    """ 
    function supports for GET request 
    """
    try:
        r = requests.get(url, timeout=3600)
        r.raise_for_status()
        res_json = r.json()
        logging.info(res_json)
        if r.status_code == 200:
            return True
        else:
            logging.warning('{\'status_code\': %s}', str(r.status_code))
            return False
    except requests.exceptions.HTTPError as error:
        logging.error(error)
        return False


def stop_server(server_port):
    """ 
    function to support server 
    """
    url = "http://127.0.0.1:" + str(server_port) + "/destroy"
    try:
        r = requests.delete(url, timeout=60)
        logging.info(r.status_code)
    except requests.ConnectionError as error:
        logging.error(error)


def clean_up(server_port, pp):
    """Clean up all resources"""
    stop_server(server_port)
    pp.communicate()
    if os.path.isfile(LLM_FILE_TO_SAVE):
        os.remove(LLM_FILE_TO_SAVE)
    if os.path.isfile(EMBED_FILE_TO_SAVE):
        os.remove(EMBED_FILE_TO_SAVE)
    with open('./test.log', 'r', encoding='utf8') as f:
        print(f.read())


def load_chat_model(model_config, path, server_port, pp):
    """Load chat model with configs from model_config"""
    new_data = {
        "ctx_len": CONST_CTX_SIZE,
        "system_prompt": model_config.system_prompt,
        "user_prompt": model_config.user_prompt,
        "ai_prompt": model_config.ai_prompt,
        "llama_model_path": path + "/" + LLM_MODEL + '.gguf',
        "model_alias": LLM_MODEL,
        "ngl": 33,
        "caching_enabled": True
    }

    url_post = "http://127.0.0.1:" + str(server_port) + "/loadmodel"

    res = request_post(new_data, url_post)
    if not res:
        clean_up(server_port, pp)
        exit(1)


def perform_chat_completion(model_config, server_port, pp):
    """Perform chat completion with model"""
    content = "How are you today?"
    user_msg = {
        "role": CONST_USER_ROLE,
        "content": content
    }
    logging.info('{\'user\': \'%s\'}', content)

    chat_data.append(user_msg)
    new_data = {
        "frequency_penalty": 0,
        "max_tokens": CONST_CTX_SIZE,
        "messages": chat_data,
        "model": LLM_MODEL,
        "presence_penalty": 0,
        "stop": model_config.stop,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95
    }

    url_post = "http://127.0.0.1:" + str(server_port) + "/v1/chat/completions"

    res = request_post(new_data, url_post, True)
    if not res:
        clean_up(server_port, pp)
        exit(1)

    content = "Tell me a short story"
    user_msg = {
        "role": CONST_USER_ROLE,
        "content": content
    }
    logging.info('{\'user\': \'%s\'}', content)

    chat_data.append(user_msg)

    new_data = {
        "frequency_penalty": 0,
        "max_tokens": CONST_CTX_SIZE,
        "messages": chat_data,
        "model": LLM_MODEL,
        "presence_penalty": 0,
        "stop": model_config.stop,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95
    }

    res = request_post(new_data, url_post, True)
    if not res:
        clean_up(port, pp)
        exit(1)


def request_embeddings(server_port, pp):
    """Get embedding data"""
    new_data = {
        "input": "This PDFs was created using Microsoft Word using the print to PDF function. True PDFs consist of both text and images. We should think about these PDFs having two layers – one layer  is the image and a second layer is the text. The image layer shows what the document will look  like if it is printed to paper. The text layer is searchable text that is carried over from the original Word file into the new PDF file (the technical term for this layer is “extracted text”). There is no need to make it searchable and the new PDF will have the same text as the original Word file. An example of True PDFs that federal defenders and CJA panel attorneys will be familiar with are the pleadings filed in CM/ECF. The pleading is originally created in Word, but then the attorney either saves it as PDF or prints to PDF and they file that PDF document with the court. Using either process, there is now a PDF file created with an image layer plus text layer. In terms of usability, this is the best type of PDF to receive in discovery as it will have the closest to text searchability of the original file",
        "model": LLM_MODEL,
        "encoding_format": "float"
    }
    url_post = "http://127.0.0.1:" + str(server_port) + "/v1/embeddings"
    res = request_post(new_data, url_post)
    if not res:
        clean_up(server_port, pp)
        exit(1)


def unload_model(model, server_port, pp):
    """Unload model"""
    new_data = {
        "model": model,
    }

    url_post = "http://127.0.0.1:" + str(server_port) + "/unloadmodel"

    res = request_post(new_data, url_post)
    if not res:
        clean_up(server_port, pp)
        exit(1)


def load_embedding_model(path, server_port, pp):
    """Load embedding model"""
    new_data = {
        "model_path": path + "/" + EMBED_MODEL + '.gguf',
        "model_alias": EMBED_MODEL,
        "ctx_len": 512,
        "ngl": 99,
        "n_parallel": 1,
        "embedding": True,
        "model_type": "embedding",
        "caching_enabled": False
    }

    url_post = "http://127.0.0.1:" + str(server_port) + "/loadmodel"

    res = request_post(new_data, url_post)
    if not res:
        clean_up(server_port, pp)
        exit(1)


def download_file(download_url, file_to_save, server_port, pp):
    """Download file with url and save to a new file"""
    if not os.path.isfile(file_to_save):
        print("Download llm model: ", download_url)
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(file_to_save, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            print("Downloaded llm model: ", download_url)
        except requests.exceptions.HTTPError as error:
            print("Had error: " + error)
            clean_up(server_port, pp)
            exit(1)
    else:
        print("Model has already existed")


port = random.randint(10000, 11000)

cwd = os.getcwd()
print(cwd)
p = subprocess.Popen([cwd + '/' + BINARY_PATH, '127.0.0.1', str(port)])
print("Server started!")

for mc in model_configs:
    logging.info('{\'Model\': \'%s\'}', mc.name)
    download_file(mc.hf_url, LLM_FILE_TO_SAVE, port, p)
    load_chat_model(mc, cwd, port, p)
    perform_chat_completion(mc, port, p)
    unload_model(LLM_MODEL, port, p)
    # Remove old data before starting with new model
    os.remove(LLM_FILE_TO_SAVE)

clean_up(port, p)

