# LLMs as Stratification Signals for KG Accuracy Evaluation
Knowledge Graph (KG) accuracy assessment is essential for ensuring data quality in downstream applications, yet remains prohibitively expensive due to annotation costs and scale. 
Large Language Models (LLMs), having been trained on very large corpora spanning a wide variety of facts, seem ideal to get access to cheap fact validation. However, their inherent hallucinations and knowledge gaps render them unreliable as direct accuracy estimators.
We propose a novel approach that exploits LLM capabilities without relying on their correctness: using aggregated LLM predictions as stratification signals for sampling-based accuracy estimation. 
By partitioning KGs into internally homogeneous strata guided by aggregated  LLM outputs, we achieve statistically significant cost reductions ranging  from 11% to 54% over unstratified and topology-based baselines on real-world  KGs. To scale beyond LLM computational constraints, we introduce a knowledge distillation strategy that transfers stratification signals to efficient student models, requiring annotation of only 0.25% of facts while maintaining signal quality. Experiments on six KGs spanning 20M+ triples demonstrate consistent improvements over SotA methods, with statistical guarantees on accuracy estimates.

## Contents
This repository contains code and data for the paper "LLMs as Stratification Signals for KG Accuracy Evaluation" submitted to VLDB 2026.

## Installation

1. Install Python 3.10. <br>
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets used in the experiments are NELL, DBPEDIA, FACTBENCH, and YAGO4.5 (noisy). <br>
Each dataset is available under: ```/dataset/{NELL|DBPEDIA|FACTBENCH|YAGO4.5}/```.

### YAGO4.5 (Noisy Versions)

The noisy variants (HQ, MQ, LQ) must be first generated. <br>
1. Follow instructions in ```/dataset/YAGO4.5/raw/README.md``` to download the original YAGO4.5 KG. <br>
2. Then, navigate to ```/dataset/YAGO4.5/```.
3. Generate a noisy dataset with: ```python generate_noisy_yago.py --accLevel {0.75|0.5|0.25}```.

This produces YAGO-HQ, YAGO-MQ, and YAGO-LQ, respectively.

## Running Experiments

### Real-World Benchmarks
For KG accuracy evaluation experiments, use the corresponding script for each dataset.

```bash
bash run_nell.sh
bash run_dbpedia.sh
bash run_factbench.sh
```

To manually configure experiments, use:

```bash
python run_eval.py --help
```

For LLM performance as direct accuracy estimators, run:

```bash
python model_eval.py --dataset {dataset_name}
```

### Scalability (Synthetic Large-Scale Benchmarks)

First, generate distillation training sets:

```bash
python verify_facts.py --dataset {YAGO4.5_0.25|YAGO4.5_0.5|YAGO4.5_0.75} --model model_name --sample True
```
**Note:** Azure API keys and inference endpoints must be configured to run LLMs.

Then, train distilled models: 

```bash
python distillation_train.py --dataset YAGO4.5 --accLevel {0.75|0.5|0.25} --targetLLM {model_name}
```

After, run distilled models over KGs to obtain predictions:

```bash
python distillation_infer.py --dataset YAGO4.5 --accLevel {0.75|0.5|0.25} --targetLLM {model_name}
```

Once (distilled) predictions are obtained, run KG accuracy evaluation with:

```bash
python run_eval.py --dataset YAGO4.5 --accLevel {0.75|0.5|0.25}
```

and assess the performance of distilled models as direct accuracy estimators with: 

```bash
python model_eval.py --dataset YAGO4.5 --accLevel {0.75|0.5|0.25}
```

### Template Prompts
Template prompts used for LLM fact verification are available in ```/template-prompts```.