#!/bin/bash

# path to evaluation procedure
python_file="./run_eval.py"

# unstratified and topology baselines
echo "run TWCS"
python "$python_file" --dataset DBPEDIA --method TWCS --stageTwoSize 3
echo "run STWCS with 'size' stratification feature"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature size

# LLM-aggregated solution
echo "run STWCS with 'llm-agg' stratification feature"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature llm-agg

# single-LLM solutions
echo "run STWCS with 'cohere' stratification feature"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature llm --targetLLM cohere
echo "run STWCS with 'deepseek-v3' stratification feature"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature llm --targetLLM deepseek-v3
echo "run STWCS with 'llama3.1-8b-it' stratification feature"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature llm --targetLLM llama3.1-8b-it
echo "run STWCS with 'mistral-nemo' stratification feature"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature llm --targetLLM mistral-nemo

# random and oracle references 
echo "run STWCS with 'random' stratification feature"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature random
echo "run STWCS with 'acc' stratification feature (oracle)"
python "$python_file" --dataset DBPEDIA --method STWCS --stageTwoSize 3 --numStrata 10 --stratFeature acc
