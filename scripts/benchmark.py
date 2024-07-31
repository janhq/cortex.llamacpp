import aiohttp
import requests
import asyncio
import time
import os
import random

start_time = time.time()
SERVER_ENDPOINT = "http://localhost:3928"
TOTAL_REQUESTS = 16
N_PARALLEL = 4
MAX_CTX_FOR_ONE_SEQUENCE = 512
N_CTX = MAX_CTX_FOR_ONE_SEQUENCE*N_PARALLEL # this number related to reserve GPU memory for kv cache

def start_server():
    import subprocess

    subprocess.run(["cd", "../examples/server/build/", "&&", "./server"])
    time.sleep(2)


def load_model():
    headers = {"Content-Type": "application/json"}
    data = {"llama_model_path": "/mnt/nas/gguf-models/meta-llama3.1-8b-instruct-q4km.gguf", "model_alias": "meta-llama3.1-8b-instruct",
            "model": "meta-llama3.1-8b-instruct", "ctx_len": N_CTX,"n_batch":2048, "ngl": 300, "model_type": "llm", "n_parallel": N_PARALLEL}
    result = requests.post(SERVER_ENDPOINT+"/loadmodel",
                           headers=headers, json=data)
    print(result.json())


async def send_request(session, prompt):
    headers = {"Content-Type": "application/json"}
    data = {"model": "meta-llama3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt},]}
    async with session.post(SERVER_ENDPOINT+"/v1/chat/completions", headers=headers, json=data) as resp:
        result = await resp.json()
        return result


async def send_request_sequence():
    # warm up
    async with aiohttp.ClientSession() as session:
        res = await send_request(session, "What is GPU?")

    start = time.time()
    total_token_processed = 0
    async with aiohttp.ClientSession() as session:

        tasks = []
        prompts = ["What is GPU?", "Who won the world cup 2022?", "Tell me some dad's joke",
                   "Write a quick sort function", "What is the price of Nvidia H100?", "Who won the world series in 2020?"]
        for number in range(TOTAL_REQUESTS):
            res = await send_request(session, random.choice(prompts))
            if res.get("usage"):
                total_token_processed += res["usage"]["total_tokens"]
            else:
                print(res)
            
    end = time.time()
    print("Finished in", end-start, "s")
    print("Total token:", total_token_processed)
    print("Throughput when run in sequence:", total_token_processed/(end-start), "tokens/s")
    print("------------------------------------------------------------------------")


async def main():
    # warm up
    async with aiohttp.ClientSession() as session:
        res = await send_request(session, "What is GPU?")

    start = time.time()
    total_token_processed = 0
    async with aiohttp.ClientSession() as session:

        tasks = []
        prompts = ["What is GPU?", "Who won the world cup 2022?", "Tell me some dad's joke",
                   "Write a quick sort function", "What is the price of Nvidia H100?", "Who won the world series in 2020?"]
        for number in range(TOTAL_REQUESTS):
            tasks.append(asyncio.ensure_future(
                send_request(session, random.choice(prompts))))

        results = await asyncio.gather(*tasks)
        for res in results:
            # print(res)
            if res.get("usage"):
                total_token_processed += res["usage"]["total_tokens"]
            else:
                print(res)
    end = time.time()
    print("Finished in", end-start, "s")
    print("Total token:", total_token_processed)
    print("Throughput when run parallel:", total_token_processed/(end-start), "tokens/s")
    print("------------------------------------------------------------------------")
# start_server()
load_model()

asyncio.run(main())

asyncio.run(send_request_sequence())
print("--- %s seconds ---" % (time.time() - start_time))
