import asyncio

from pipeline.verification import verify
from argparse import ArgumentParser


if __name__ == '__main__':
    # argument parsing
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBPEDIA', choices=['DBPEDIA', 'FACTBENCH', 'NELL', 'YAGO4.5_0.25', 'YAGO4.5_0.5', 'YAGO4.5_0.75'], help='The KG to be evaluated.')
    parser.add_argument('--model', type=str, default='cohere', choices=['cohere', 'deepseek-v3', 'llama3.1-8b-it', 'mistral-nemo'], help='The LLM to be run.')
    parser.add_argument('--max_retries', type=int, default=5, help='The maximum number of retries the LLM is allowed to take.')
    parser.add_argument('--key', type=str, default='', help='The Azure API key required to access the models.')
    parser.add_argument('--endpoint', type=str, default= '', help='The Azure AI Inference endpoint.')
    parser.add_argument('--max_reqs', type=int, default=30, help='Maximum number of concurrent requests to Azure API.')
    parser.add_argument('--sample', type=bool, default=False, help='Whether to verify all KG facts (False) or a sample (True). When sample=True, the function is used to generate distillation training sets.')
    parser.add_argument('--sample_size', type=int, default=50000, help='Sample size for the distillation training set. This parameter applies only when sample=True.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    # run verification pipeline
    asyncio.run(verify(args))
