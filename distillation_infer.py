import argparse
import torch
import json
import os

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pipeline.llm.format import format_rdf_item


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='YAGO4.5', choices=['YAGO4.5'], help='Target dataset.')
parser.add_argument('--accLevel', default=0.5, choices=[0.25, 0.5, 0.75], type=float, help='Accuracy level for large-scale, synthetic KG.')
parser.add_argument('--targetLLM', default='cohere', choices=['', 'cohere', 'llama3.1-8b-it', 'mistral-nemo', 'deepseek-v3'], help='Target LLM.')
parser.add_argument('--per_device_batch_size', default=64, type=int, help='Per-device batch size.')
args = parser.parse_args()


# utils
def get_available_gpus():
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


# custom dataset class
class KGDataset(Dataset):
    """
    Torch dataset for KG triples.
    """

    def __init__(self, kg, kg_name, tokenizer, device, max_length=512):
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.ids = []
        self.triples = []
        for triple_id, triple in kg.items():
            subject, predicate, obj = self._format_triple(triple, kg_name)
            triple_text = f"{subject} [SEP] {predicate} [SEP] {obj}"

            self.ids.append(triple_id)
            self.triples.append(triple_text)

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
        return len(self.ids)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.triples[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': enc['input_ids'].squeeze().to(self.device),
            'attention_mask': enc['attention_mask'].squeeze().to(self.device),
            'id': self.ids[idx]
        }


def main():
    # fetch GPUs
    gpus = get_available_gpus()
    print(f"Found GPUs: {gpus}")
    
    # set device
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # set KG path
    dataset = args.dataset
    if args.dataset == 'YAGO4.5':
        dataset += f'/{args.accLevel}'

    # load KG
    kg_file = f'./dataset/{dataset}/data/kg.json'
    print('Loading {} dataset'.format(dataset))
    with open(kg_file, 'r') as f:
        kg = json.load(f)
    print('{} dataset loaded!'.format(dataset))

    # load tokenizer and model
    model_path = f'./distilled_models/{dataset}/{args.targetLLM}'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model = torch.nn.DataParallel(model, device_ids=gpus) if len(gpus) > 1 else model  # parallelize if possible
    model.to(device)
    model.eval()

    # dataset and dataloader
    dataset = KGDataset(kg, args.dataset, tokenizer, device)
    total_batch_size = args.per_device_batch_size * max(1, len(gpus))
    dataloader = DataLoader(dataset, batch_size=total_batch_size)

    idx2label = {0: 'false', 1: 'true'}
    gtLLM = {}

    # run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting triple correctness'):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            for triple_id, pred in zip(batch['id'], preds):
                gtLLM[triple_id] = idx2label[pred]

    # setup path where storing predictions
    save_path = f"./predictions/{args.dataset}/full/{args.targetLLM}/"
    os.makedirs(save_path, exist_ok=True)

    # store predictions
    print(f'Save {args.targetLLM}-distilled predictions to {save_path}')
    with open(save_path + 'preds.json', 'w') as out:
        json.dump(gtLLM, out)
    print(f'Predictions saved to {save_path}')


if __name__ == '__main__':
    main()
