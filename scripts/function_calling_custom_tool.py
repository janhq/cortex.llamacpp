import requests
import time
import os
import random
import json
start_time = time.time()
SERVER_ENDPOINT = "http://localhost:3928"
TOTAL_USERS = 1
NUM_ROUNDS = 10
MAX_TOKENS = 500
N_PARALLEL = 1
MAX_CTX_FOR_ONE_SEQUENCE = 500
# this number related to reserve GPU memory for kv cache
N_CTX = MAX_CTX_FOR_ONE_SEQUENCE*N_PARALLEL
import re
import json

def function_detect_function(response:str):
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)
    return match

def parse_tool_response(response: str):
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)

    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {
                "name": function_name,
                "parameters": args,
            }
        except json.JSONDecodeError as error:
            print(f"Error parsing function arguments: {error}")
            return None
    return None

def load_model():
    headers = {"Content-Type": "application/json"}
    data = {"llama_model_path": "/mnt/nas/gguf-models/meta-llama3.1-8b-instruct-q4km.gguf", "model_alias": "meta-llama3.1-8b-instruct", "engine": "cortex.llamacpp",
            "model": "meta-llama3.1-8b-instruct", "ctx_len": N_CTX, "ngl": 300, "model_type": "llm", "n_parallel": N_PARALLEL}

    result = requests.post(SERVER_ENDPOINT+"/loadmodel",
                           headers=headers, json=data)
    # result = requests.post(SERVER_ENDPOINT+"/inferences/server/loadmodel",
    print(result.json())


def send_request(data):
    stream = False
    headers = {"Content-Type": "application/json"}

    result = requests.post(
        SERVER_ENDPOINT+"/v1/chat/completions", headers=headers, json=data)

    return result.json()


weatherTool = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
}

toolPrompt = f"""

Environment: ipython
Tools: brave_search, wolfram_alpha

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search


Access to the following functions:

Use the function '{weatherTool["name"]}' to: '{weatherTool["description"]}'
{json.dumps(weatherTool)}

If you choose to call a function ONLY reply in the following format:

<function=example_function_name>{{\"example_name\": \"example_value\"}}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query

You are a helpful assistant.

"""

messages = [
    {
        "role": "system",
        "content": toolPrompt,
    },
    {
        "role": "user",
        "content": "What is the weather in Tokyo?",
    },

]


data = {"model": "meta-llama3.1-8b-instruct", "max_tokens": 200, "stop": ["<|eom_id|>", "<|end_of_text|>", "<|eot_id|>"], "engine": "cortex.llamacpp", "stream": False,
        "messages": messages}

load_model()
function_ = send_request(data)["choices"][0]["message"]["content"]

print(function_)
function_detect = function_detect_function(function_) #function_.split("ASSISTANT:")[0]
print(function_detect.span())
position_match = function_detect.span()
function_detect = function_[position_match[0]: position_match[1]]

parsed_response= parse_tool_response(function_)

def get_current_weather(location: str) -> str:
    # This would be replaced by a weather API
    if location == "San Francisco, CA":
        return "62 degrees and cloudy"
    elif location in "Tokyo, JP" or location in "Tokyo, Japan":
        return "70 degrees and rain"
    return "Weather is unknown"


available_functions = {"get_current_weather": get_current_weather}
function_to_call = available_functions[parsed_response["name"]]
weather = function_to_call(parsed_response["parameters"]["location"])

messages.append({
    "role":"assistant",
    "content": function_detect
})
messages.append({"role":"ipython","content":weather})

print("Weather answer is: ", weather)
data = {"model": "meta-llama3.1-8b-instruct", "max_tokens": MAX_TOKENS, "stop": ["<|eom_id|>", "<|end_of_text|>", "<|eot_id|>"], "engine": "cortex.llamacpp", "stream": False,
        "messages": messages}

res = send_request(data)

print("Answer from the LLM: ", res["choices"][0]["message"])
