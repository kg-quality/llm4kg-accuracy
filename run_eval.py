import pandas as pd
import numpy as np
import argparse
import random
import json
import os

from glob import glob
from scipy import stats
from framework import samplingMethods

parser = argparse.ArgumentParser()

################################
###### Dataset parameters ######
################################

parser.add_argument('--dataset', default='NELL', choices=['DBPEDIA', 'FACTBENCH', 'NELL', 'YAGO4.5'], help='Target dataset.')
parser.add_argument('--accuracyLevel', default=0.25, choices=[0.25, 0.5, 0.75], type=float, help='Desired KG accuracy level. Use only when working with YAGO4.5 datasets.')

###############################
###### Method parameters ######
###############################

parser.add_argument('--method', default='STWCS', choices=['TWCS', 'STWCS'], help='Method of choice.')
parser.add_argument('--minSample', default=30, type=int, help='Min sample size required to perform eval.')
parser.add_argument('--stageTwoSize', default=3, type=int, help='Second-stage sample size. Required by two-stage sampling methods.')
parser.add_argument('--ciMethod', default='bayesHPD', choices=['bayesHPD'], help='Methods to construct intervals.')
parser.add_argument('--alphaPrior', default=-1, type=float, help='Parameter alpha used to setup Beta prior distribution -- only applies to bayesian credible intervals.')
parser.add_argument('--betaPrior', default=-1, type=float, help='Parameter beta used to setup Beta prior distribution -- only applies to bayesian credible intervals.')

#######################################
###### Stratification parameters ######
#######################################

parser.add_argument('--numStrata', default=10, type=int, help='Number of strata considered by stratification based sampling methods.')
parser.add_argument('--stratFeature', default='llm-agg', choices=['acc', 'llm', 'llm-agg', 'random', 'size'], help='Stratification feature of choice. Note that "acc" represents the oracle stratification feature (use to estimate lower bound only).')
parser.add_argument('--usePreds', default=True, type=bool, help='Whether to consider LLM predictions for stratification.')
parser.add_argument('--targetLLM', default='', choices=['', 'cohere', 'llama3.1-8b-it', 'mistral-nemo', 'deepseek-v3'], help='Target LLM. Use only when stratifying w/ a single LLM, otherwise set to empty string.')

###################################
###### Estimation parameters ######
###################################

parser.add_argument('--confLevel', default=0.05, type=float, help='Estimator confidence level (1-confLevel).')
parser.add_argument('--thrMoE', default=0.05, type=float, help='Threshold for Margin of Error (MoE).')

###################################
###### Annotation parameters ######
###################################

parser.add_argument('--c1', default=45, type=int, help='Average cost for Entity Identification (EI).')
parser.add_argument('--c2', default=25, type=int, help='Average cost for Fact Verification (FV).')

##################################
###### Computing parameters ######
##################################

parser.add_argument('--iterations', default=1000, type=int, help='Number of iterations for computing estimates.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

args = parser.parse_args()


def samplingMethod(method, confLevel, alphaPrior, betaPrior):
    """
    Instantiate the specific sampling method

    :param method: sampling method
    :param confLevel: estimator confidence level (1-confLevel)
    :param alphaPrior: alpha parameter used for Beta prior distribution (only applies to bayesian credible intervals)
    :param betaPrior: beta parameter used for Beta prior distribution (only applies to bayesian credible intervals)
    :return: instance of specified sampling method
    """

    return {
        'TWCS': lambda: samplingMethods.TWCSSampler(confLevel, alphaPrior, betaPrior),
        'STWCS': lambda: samplingMethods.STWCSSampler(confLevel, alphaPrior, betaPrior)
    }[method]()


def main():
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set the KG dataset path
    dataset_path = f'./dataset/{args.dataset}'
    if args.dataset == 'YAGO4.5':  # specify accuracy level to generate YAGO-{HQ, MQ, LQ}
        dataset_path += f'/{args.accuracyLevel}'

    # load target KG dataset
    print(f'Loading {args.dataset} dataset at {dataset_path}...')
    with open('./dataset/' + args.dataset + '/data/kg.json', 'r') as f:
        id2triple = json.load(f)
    print(f'Dataset {args.dataset} at {dataset_path} loaded!')

    # set KG as [(id, triple), ...]
    kg = list(id2triple.items())

    # load KG ground truth
    with open('./dataset/' + args.dataset + '/data/gt.json', 'r') as f:  # get ground truth
        gt = json.load(f)

    if args.usePreds:  # consider LLM predictions -- used to stratify KG facts
        preds = {}
        # get LLM prediction files
        predFs = glob('./predictions/' + args.dataset + '/**/preds.json')
        for predF in predFs:  # iterate over prediction files and store LLM preds
            with open(predF, 'r') as f:
                preds[predF.split('/')[3]] = json.load(f)
        # sanity check
        assert len(preds) != 0
    else:  # ignore LLM predictions
        preds = None

    # compute KG (real) accuracy
    acc = sum(gt.values())/len(gt)
    print('KG (real) accuracy: {}'.format(acc))

    # set efficient KG accuracy estimator w/ confidence level 1-args.confLevel
    print('Set {} estimator with confidence level {}%.'.format(args.method, 1 - args.confLevel))
    estimator = samplingMethod(args.method, args.confLevel, args.alphaPrior, args.betaPrior)

    # set params to perform evaluation
    eParams = {'kg': kg, 'groundTruth': gt, 'minSample': args.minSample, 'thrMoE': args.thrMoE, 'ciMethod': args.ciMethod, 'iters': args.iterations}
    if (args.method == 'TWCS') or (args.method == 'STWCS'):  # two-stage sampling methods require second-stage sample size parameter
        eParams['stageTwoSize'] = args.stageTwoSize
    if args.method == 'STWCS':  # stratified sampling methods require stratification features and desired number of strata
        eParams['stratFeature'] = args.stratFeature
        eParams['numStrata'] = args.numStrata
        eParams['targetLLM'] = args.targetLLM
        eParams['preds'] = preds

    eParams['c1'] = args.c1
    eParams['c2'] = args.c2

    # perform the evaluation procedure for args.iterations times and compute estimates
    print('Perform KG accuracy evaluation for {} times and stop at each iteration when MoE <= {}'.format(args.iterations, args.thrMoE))
    estimates = estimator.run(**eParams)
    # convert estimates to numpy and print results
    estimates = pd.DataFrame(estimates, columns=['annotTriples', 'estimatedAcc', 'annotCost', 'lowerBound', 'upperBound'])
    print('estimated accuracy: mean={} stdev={}'.format(estimates['estimatedAcc'].mean(), estimates['estimatedAcc'].std()))
    print('annotated triples: mean={} stdev={}'.format(estimates['annotTriples'].mean(), estimates['annotTriples'].std()))
    print('annotation cost (hours): mean={} stdev={}'.format(estimates['annotCost'].mean(), estimates['annotCost'].std()))

    # create dir (if not exists) where storing estimates
    dname = './results/' + args.dataset + '/alpha=' + str(args.confLevel) + '/'
    dname += args.ciMethod + '/a=' + str(round(args.alphaPrior, 2)) + '_b=' + str(round(args.betaPrior, 2)) + '/'
    os.makedirs(dname, exist_ok=True)
    # set file name
    fname = args.method + '_stage2=' + str(args.stageTwoSize)
    if args.method == 'STWCS':
        fname += '_feature=' + args.stratFeature
        if args.stratFeature == 'llm' and args.targetLLM:
            fname += '-' + args.targetLLM
        fname += '_strata=' + str(args.numStrata)
    # add file extension
    fname += '.tsv'
    # store estimates
    estimates.to_csv(dname+fname, sep='\t', index=False)
    print('Estimates stored in {}{}'.format(dname, fname))


if __name__ == "__main__":
    main()
