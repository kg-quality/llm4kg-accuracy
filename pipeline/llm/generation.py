import asyncio

from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter


# limit to 500 requests per minute (60 seconds) -- conservative bound to manage both requests-per-minute and tokens-per-minute
RATE_LIMITER = AsyncLimiter(500, 60)

async def limited_generate(chat, llm, _id, semaphore):
    """
    Wrap the LLM call on Azure w/ concurrency control (semaphore) and rate limiting (limiter).

    :param chat: the user-LLM chat
    :param llm: the considered LLM
    :param _id: the ID of the fact associated w/ the chat
    :param semaphore: the concurrency semaphore
    :return: the fact ID w/ the LLM veracity prediction
    """

    async with semaphore:  # limits concurrent entries
        async with RATE_LIMITER:
            response = await generate_response_async(chat, llm)
            return _id, response


async def generate_response_async(chat, llm, max_tokens=1):
    """
    Send a chat prompt to an LLM on Azure and return its text response (async).

    :param chat: the user-LLM chat
    :param llm: the considered LLM
    :param max_tokens: max tokens allowed in the response
    :return: the LLM response
    """

    # call the Azure inference API to fetch the LLM response for the provided chat
    response = await llm.complete(
        messages=chat,
        max_tokens=max_tokens,
        temperature=0.0,  # deterministic
        top_p=1.0,  # disables nucleus sampling
        read_timeout=10  # the time you are willing to wait for some data to be received from the connected party
    )
    # return text response
    return response.choices[0].message.content


async def batch_generate_async(chats, llm, max_concurrency=300):
    """
    Process batch of chats concurrently with controlled parallelism (async) and rate limiting.

    :param chats: the user-LLM chats
    :param llm: the considered LLM
    :param max_concurrency: max number of concurrent requests -- set to the Azure rate of 300
    :return: the fact IDs w/ corresponding LLM veracity predictions (dict)
    """

    # set up the concurrency semaphore
    semaphore = asyncio.Semaphore(max_concurrency)
    # set routine for each chat
    tasks = [limited_generate(chat, llm, _id, semaphore) for _id, chat in chats.items()]

    preds = {}
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Verifying facts"):  # process tasks
        _id, response = await future
        preds[_id] = response
    return preds
