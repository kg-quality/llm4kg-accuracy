import numpy as np
import argparse
import random
import torch
import json

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, set_seed
from torch.utils.data import Dataset

from pipeline.llm.format import format_rdf_item


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='YAGO4.5', choices=['YAGO4.5'], help='Target dataset.')
parser.add_argument('--accLevel', default=0.25, choices=[0.25, 0.5, 0.75], type=float, help='Accuracy level for large-scale, synthetic KG.')
parser.add_argument('--targetLLM', default='cohere', choices=['', 'cohere', 'llama3.1-8b-it', 'mistral-nemo', 'deepseek-v3'], help='Target LLM.')
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
args = parser.parse_args()


# custom dataset class
class KGDataset(Dataset):
    """
    Torch dataset for KG triples w/ LLM pseudo-labels.
    """

    def __init__(self, kg, kg_name, preds, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []
        self.labels = []

        for triple_id, triple in kg.items():
            subject, predicate, obj = self._format_triple(triple, kg_name)
            triple_text = f"{subject} [SEP] {predicate} [SEP] {obj}"

            self.samples.append(triple_text)
            self.labels.append(preds[triple_id])

    @staticmethod
    def _format_triple(triple, kg_name):
        if kg_name == 'DBPEDIA':
            return [str(item).split('/')[-1].replace('_', ' ') for item in triple]
        if kg_name == 'NELL':
            return [str(item).split(':')[-1].replace('_', ' ') for item in triple]
        if kg_name == 'YAGO4.5':
            return [format_rdf_item(item) for item in triple]
        return triple

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.samples[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def main():
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    # deterministic GPU behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # label mapping
    label2idx = {'false': 0, 'true': 1}

    # set KG and prediction paths
    dataset = args.dataset
    if args.dataset == 'YAGO4.5':
        dataset += f'/{args.accLevel}'
    kg_file = f'./dataset/{dataset}/data/kg.json'
    pred_file = f'./predictions/{dataset}/sample/{args.targetLLM}/preds.json'

    # load KG and predictions
    print('Loading {} dataset...'.format(dataset))
    with open(kg_file, 'r') as f:
        kg = json.load(f)
    print('{} dataset loaded!'.format(dataset))

    with open(pred_file, 'r') as f:
        id2label = json.load(f)  # LLM predictions are stored as {factID: label, ...}
    preds = {id_: label2idx[label] for id_, label in id2label.items() }

    # restrict KG to the sample of triples w/ predictions
    kg = {id_: triple for id_, triple in kg.items() if id_ in preds}
    print(f"Loaded {len(kg)} triples w/ predictions.")

    # setup tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = KGDataset(kg, args.dataset, preds, tokenizer)

    # setup model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # training arguments
    training_args = TrainingArguments(
        output_dir="./unused_output",
        dataloader_pin_memory=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="no",
        save_strategy="no",
        logging_dir=None,
        report_to="none",
        no_cuda=False,  # explicit CUDA usage (if available)
    )

    # setup trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    print(f"Training on {args.dataset} dataset ({len(dataset)} samples)...")
    trainer.train()

    # save trained model + tokenizer
    save_path = f"./distilled_models/{dataset}/{args.targetLLM}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


if __name__ == '__main__':
        main()
