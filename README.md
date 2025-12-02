# LLMs as Stratification Signals for KG Accuracy Evaluation
Knowledge Graph (KG) accuracy assessment is essential for ensuring data quality in downstream applications, yet remains prohibitively expensive due to annotation costs and scale. 
Large Language Models (LLMs), having been trained on very large corpora spanning a wide variety of facts, seem ideal to get access to cheap fact validation. However, their inherent hallucinations and knowledge gaps render them unreliable as direct accuracy estimators.
We propose a novel approach that exploits LLM capabilities without relying on their correctness: using aggregated LLM predictions as stratification signals for sampling-based accuracy estimation. 
By partitioning KGs into internally homogeneous strata guided by aggregated  LLM outputs, we achieve statistically significant cost reductions ranging  from 11% to 54% over unstratified and topology-based baselines on real-world  KGs. To scale beyond LLM computational constraints, we introduce a knowledge distillation strategy that transfers stratification signals to efficient student models, requiring annotation of only 0.25% of facts while maintaining signal quality. Experiments on six KGs spanning 20M+ triples demonstrate consistent improvements over SotA methods, with statistical guarantees on accuracy estimates.

## Contents
This repository contains code and data for the paper "LLMs as Stratification Signals for KG Accuracy Evaluation" submitted to VLDB 2026.

## Installation

Install Python 3.10 if not already installed.

## Running Experiments

For each dataset, experiments can be executed using the corresponding script.

To run experiments on NELL, run:

```bash
bash run_nell.sh
```

To run experiments on DBPEDIA, run:

```bash
bash run_dbpedia.sh
```

To run experiments on FACTBENCH, run:

```bash
bash run_factbench.sh
```

To experiment with different configurations, directly use ```run_eval.py```. The description of all the available arguments can be obtained by running:

```bash
python run_eval.py --help
```
