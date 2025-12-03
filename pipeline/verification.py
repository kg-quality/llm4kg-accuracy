import json
import random

from pathlib import Path

from llm.models import model2id
from llm.generation import batch_generate_async
from llm.format import prepare_fact, convert_response, check_response

from prompts.compose import compose_prompts

from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import UserMessage, SystemMessage, AssistantMessage


async def verify(args):
    """
    Perform (asynchronous) verification on KG facts w/ a given LLM.

    :param args: argument parser arguments
    :return: none
    """

    # set seed
    random.seed(args.seed)

    # get the prompts
    is_dbpedia = (args.dataset == 'DBPEDIA')
    full_kwargs = {
        'max_retries': args.max_retries,
        'dataset': args.dataset
    }

    # compose prompt templates
    system_prompt, user_prompt, retry_prompt = compose_prompts(is_dbpedia, **full_kwargs)

    # set the path to dataset
    if 'YAGO4.5' in args.dataset:  # YAGO4.5 comes in three versions -- preprocess string to extract the target version
        yago_ver = args.dataset.split('_')
        data_path = f'./dataset/{yago_ver[0]}/{yago_ver[1]}/data/kg.json'
    else:
        data_path = f'./dataset/{args.dataset}/data/kg.json'

    # load dataset
    with open(data_path) as handler:
        kg = json.load(handler)

    if args.sample:  # sample facts from KG -- LLM predictions are used to train distilled models
        sample_ids = random.sample(list(kg.keys()), args.sample_size)  # draw sample_size fact IDs w/o replacement
        kg = {factID: kg[factID] for factID in sample_ids}  # restrict KG to sample

    # set var to store LLM predictions (LLM ground-truth)
    gtLLM = {}

    # prompt LLM to verify facts within the KG dataset
    print('Verifying KG facts...')
    print(
        '\nConfig:\n'
        f'- model: {args.model}\n'
        f'- dataset: {args.dataset}\n'
    )

    chats = {}
    fact2prep = {}
    for factID, fact in kg.items():  # prepare chats for asynchronous processing
        # preprocess facts for prompt injection -- obtain a dict with keys (s, p, o)
        prepared_fact = prepare_fact(fact, args.dataset)
        fact2prep[factID] = prepared_fact
        # system and user prompts are concatenated to provide all models with the same prompt structure
        chats[factID] = [SystemMessage(content=system_prompt), UserMessage(content=user_prompt.format(**prepared_fact))]

    # setup client for asynchronous chat completion
    async with ChatCompletionsClient(
            endpoint=args.endpoint,
            credential=AzureKeyCredential(args.key),
            model=model2id[args.model],
    ) as llm:
        # prompt LLM to generate veracity responses over fact-related prompts
        responses = await batch_generate_async(chats, llm, max_concurrency=args.max_reqs)

        # setup var to store number of retries
        retries = 0
        missing = {}  # dict where storing LLM responses that diverge from the allowed ones
        for factID, response in responses.items():  # iterate over LLM responses and store them
            if check_response(response):  # LLM provided proper response -- store response and remove fact from chats
                gtLLM[factID] = convert_response(response)
                del chats[factID]
            else:  # LLM did not provide proper response -- store improper response as missing
                missing[factID] = response

        while missing:  # if the LLM response diverges from the allowed ones, ask again
            if retries == args.max_retries:  # reached max number of retries -- store NA for the remaining missing facts
                for factID in missing.keys():
                    gtLLM[factID] = convert_response('NA')
                    missing = []
            else:  # prompt the LLM again updating the number of attempts
                attempts = args.max_retries - retries
                for factID, response in missing.items():  # for each missing fact, update the corresponding prompt w/ the retry mechanism
                    chats[factID] += [AssistantMessage(content=response), UserMessage(content=retry_prompt.format(attempts=attempts, **fact2prep[factID]))]
                # re-prompt LLM to generate veracity responses over fact-related prompts
                responses = await batch_generate_async(chats, llm, max_concurrency=args.max_reqs)

                for factID, response in responses.items():  # iterate over LLM responses and store them
                    if check_response(response):  # LLM provided proper response -- store response and remove fact from chats and missing
                        gtLLM[factID] = convert_response(response)
                        del chats[factID]
                        del missing[factID]
                    else:  # LLM did not provide proper response -- update improper response in missing
                        missing[factID] = response
                # update the number of performed retries
                retries += 1
    print(f'KG facts verified!')

    # setup folder to store predictions
    if args.sample:  # store predictions within 'sample' folder
        out_dir = 'sample'
    else:  # store predictions within 'full' folder
        out_dir = 'full'

    print('Storing LLM predictions...')
    result_path = Path(f'./predictions/{args.dataset}/{out_dir}/{args.model}')
    result_path.mkdir(parents=True, exist_ok=True)

    # store predictions in folder
    with open(result_path / 'preds.json', 'w') as handler:
        json.dump(gtLLM, handler)
    print('LLM predictions stored!')
