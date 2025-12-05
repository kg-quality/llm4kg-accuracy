import json
import argparse

from glob import glob
from sklearn.metrics import balanced_accuracy_score, f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NELL', choices=['DBPEDIA', 'FACTBENCH', 'NELL', 'YAGO4.5'], help='Target dataset.')
parser.add_argument('--accLevel', default=0.25, choices=[0.25, 0.5, 0.75], type=float, help='Desired KG accuracy level. Use only when working with YAGO4.5 datasets.')
args = parser.parse_args()


def main():
    # set KG path
    dataset = args.dataset
    if args.dataset == 'YAGO4.5':
        dataset += f'/{args.accLevel}'

    # load ground truth
    print(f'Load {dataset} ground truth')
    with open(f'./dataset/{dataset}/data/gt.json', 'r') as f:
        gt = json.load(f)

    # load model prediction files (LLM or distilled)
    pred_files = glob(f'./predictions/{dataset}/full/**/preds.json')

    model2preds = {}
    for pred_file in pred_files:  # iterate over prediction files and store LLM preds
        model_name = pred_file.split('/')[4]
        with open(pred_file, 'r') as f:
            model2preds[model_name] = json.load(f)

    # setup label-index map
    label2idx = {'false': 0, 'true': 1}

    for model, preds in model2preds.items():  # evaluate each model
        y_true = []
        y_pred = []

        for triple_id, true_label in gt.items():  # store true/predicted labels
            y_true.append(true_label)
            y_pred.append(label2idx[preds[triple_id]])
        # sanity check
        assert len(y_true) == len(y_pred)

        print(f'\nModel: {model}')
        # accuracy estimate
        print(f'Model accuracy estimate: mu={sum(y_pred)/len(y_pred):.2f}')
        # metrics
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f'Balanced Accuracy: {bacc:.2f}, Macro F1: {f1:.2f}')
        print('###########')


if __name__ == '__main__':
    main()
