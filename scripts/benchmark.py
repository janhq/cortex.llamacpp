import aiohttp
import requests
import asyncio
import time
import os
import random

start_time = time.time()
SERVER_ENDPOINT = "http://localhost:3928"
TOTAL_USERS = 40
NUM_ROUNDS = 10
MAX_TOKENS = 500
N_PARALLEL = 32
MAX_CTX_FOR_ONE_SEQUENCE = 1000
# this number related to reserve GPU memory for kv cache
N_CTX = MAX_CTX_FOR_ONE_SEQUENCE*N_PARALLEL


def start_server():
    import subprocess

    subprocess.run(["cd", "../examples/server/build/", "&&", "./server"])
    time.sleep(2)


def load_model():
    headers = {"Content-Type": "application/json"}
    data = {"llama_model_path": "/mnt/nas/gguf-models/meta-llama3.1-8b-instruct-q4km.gguf", "model_alias": "meta-llama3.1-8b-instruct","engine": "cortex.llamacpp",
            "model": "meta-llama3.1-8b-instruct", "ctx_len": N_CTX, "ngl": 300, "model_type": "llm", "n_parallel": N_PARALLEL}
    
    result = requests.post(SERVER_ENDPOINT+"/loadmodel",
                           headers=headers, json=data)
    # result = requests.post(SERVER_ENDPOINT+"/inferences/server/loadmodel",
    #                        headers=headers, json=data)
    print(result.json())


async def send_request(session, prompt,sleep = 0):
    await asyncio.sleep(sleep)
    headers = {"Content-Type": "application/json"}
    data = {"model": "meta-llama3.1-8b-instruct", "max_tokens": MAX_TOKENS, "stop": ["<|eom_id|>", "<|end_of_text|>", "<|eot_id|>"],"engine": "cortex.llamacpp",
            "messages": [{"role": "user", "content": prompt},]}
    async with session.post(SERVER_ENDPOINT+"/v1/chat/completions", headers=headers, json=data) as resp:
        result = await resp.json()
        return result

async def one_user(session, prompt):
    tasks = [send_request(session, prompt,random.random()*0.2+ i ) for i in range(NUM_ROUNDS)]
    results = await asyncio.gather(*tasks)
    return results


async def send_request_sequence():
    # warm up
    async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout()) as session:
        res = await send_request(session, "What is GPU?")

    start = time.time()
    total_token_processed = 0
    async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout()) as session:

        tasks = []
        prompts = ["What is GPU?", "Who won the world cup 2022?", "Tell me some dad's joke",
                   "Write a quick sort function", "What is the price of Nvidia H100?", "Who won the world series in 2020?"]
        for number in range(TOTAL_USERS):
            res = await send_request(session, random.choice(prompts))
            if res.get("usage"):
                total_token_processed += res["usage"]["total_tokens"]
            else:
                print(res)

    end = time.time()
    print("Finished in", end-start, "s")
    print("Total token:", total_token_processed)
    print("Throughput when run in sequence:",
          total_token_processed/(end-start), "tokens/s")
    print("------------------------------------------------------------------------")


async def main():
    # warm up
    async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout()) as session:
        res = await send_request(session, "What is GPU?")

    start = time.time()
    total_token_processed = 0
    async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout()) as session:

        tasks = []
        prompts = [
            "What is GPU?",
            "Who won the world cup 2022?",
            "Tell me so many dad's joke,",
            "Write a quick sort function,",
            "What is the price of Nvidia H100?",
            "Who won the world series in 2020?",
            "Tell me a very long story,",
            "Who is the best football player in the world?",
            "Tell me about compiler,",
            "Tell me about AI,"]
        for number in range(TOTAL_USERS):
            tasks.append(asyncio.ensure_future(
                one_user(session, random.choice(prompts))))

        list_results = await asyncio.gather(*tasks)
        for results in list_results:
            for res in results:
                # print(res)
                if res.get("usage"):
                    total_token_processed += res["usage"]["total_tokens"]
                else:
                    print(res)
    end = time.time()
    print("Finished in", end-start, "s")
    print("Total token:", total_token_processed)
    print("Throughput when run parallel:",
          total_token_processed/(end-start), "tokens/s")
    print("------------------------------------------------------------------------")
    with open("result.log","w") as writer:
        for results in list_results:
            for res in results:
                try:
                    writer.write(res["choices"][0]["message"]["content"] + "\n\n")
                except:
                    continue
# start_server()
load_model()

asyncio.run(main())

# asyncio.run(send_request_sequence())
print("--- %s seconds ---" % (time.time() - start_time))
