import os
import json
import random
import argparse
import lightrdf
import numpy as np
import polars as pl

from tqdm import tqdm
from lightrdf import Regex


parser = argparse.ArgumentParser()
parser.add_argument("--accLevel", default=0.25, choices=[0.25, 0.5, 0.75], type=float)
parser.add_argument("--inputYago", default="./raw/yago-facts.ttl", type=str)
parser.add_argument("--outputRoot", default="./", type=str)
args = parser.parse_args()


# YAGO predicates to filter
FILTER_PREDICATES = [
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
    "<http://www.w3.org/2000/01/rdf-schema#subPropertyOf>",
    "<http://www.w3.org/2000/01/rdf-schema#subClassOf>",
    "<http://www.w3.org/2000/01/rdf-schema#domain>",
    "<http://www.w3.org/2000/01/rdf-schema#range>",
    "<http://www.w3.org/2000/01/rdf-schema#label>",
    "<http://www.w3.org/2000/01/rdf-schema#comment>",
    "<http://schema.org/mainEntityOfPage>",
    "<http://www.w3.org/2002/07/owl#sameAs>",
    "<http://schema.org/alternateName>",
    "<http://proton.semanticweb.org/protonsys#transitiveOver>",
    "<http://www.w3.org/2002/07/owl#inverseOf>",
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
    "<http://yago-knowledge.org/schema#fromClass>",
    "<http://www.w3.org/2002/07/owl#disjointWith>",
    "<http://schema.org/image>",
    "<http://yago-knowledge.org/schema#fromProperty>",
    "<http://schema.org/url>",
    "<http://schema.org/sameAs>",
    "<http://schema.org/parentTaxon>",
    "<http://schema.org/logo>",
    "<http://schema.org/geo>",
    "<http://yago-knowledge.org/resource/consumes>",
]


def filter_yago(input_path, output_csv):
    """
    Load YAGO triples and filter by predicate and subject regex.

    :param input_path: path to YAGO TTL file
    :param output_csv: where to store filtered triples
    :return: None
    """
    print("Filtering YAGO facts...")
    kg = lightrdf.RDFDocument(input_path)

    triples = []
    for s, p, o in tqdm(kg.search_triples(Regex("^<http://yago-knowledge.org/resource/.*>$"), None, None)):
        if p not in FILTER_PREDICATES:
            triples.append((s, p, o))
    print("YAGO facts filtered!")

    df = pl.DataFrame(triples, schema=["Subject", "Relation", "Object"])
    df.write_csv(output_csv)
    print(f"Filtered YAGO facts stored in {output_csv}")


def generate_noisy_triple(triple, subjects, objects, kg_set, noisy_set):
    """
    Generate one noisy triple by corrupting subject/object or both.

    :param triple: triple to corrupt
    :param subjects: subjects per predicate
    :param objects: objects per predicate
    :param kg_set: KG triple set
    :param noisy_set: noisy triple set
    :return: corrupted triple
    """

    s, p, o = triple
    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        s_new, p_new, o_new = s, p, o
        choice = random.choice([1, 2, 3])

        # generate corrupted elements
        if choice == 1:  # corrupt subject
            s_new = random.choice(subjects[p])
        elif choice == 2:  # corrupt object
            o_new = random.choice(objects[p])
        else:  # corrupt both
            s_new = random.choice(subjects[p])
            o_new = random.choice(objects[p])

        corrupted = (s_new, p_new, o_new)

        # ensure uniqueness and avoid true triples
        if corrupted not in kg_set and corrupted not in noisy_set:
            return corrupted

        attempts += 1

    print("WARNING: Could not generate unique noisy triple")
    return None


def corrupt_triples(kg, acc_level):
    """
    Corrupt KG triples to reach the target accuracy.

    :param kg: the considered KG
    :param acc_level: the target accuracy level
    :return: corrupted KG w/ ground truth
    """

    total = kg.height
    correct_count = int(total * acc_level)
    noisy_count = total - correct_count

    print(f"KG size: {total}, correct: {correct_count}, noisy: {noisy_count}")

    # sample correct triples
    correct_df = kg.sample(n=correct_count, with_replacement=False)

    # remaining triples must be noised
    noisy_candidates = kg.join(correct_df, on=["Subject", "Relation", "Object"], how="anti")

    # build lookups for corruptions
    pred2subj = kg.group_by("Relation").agg(pl.col("Subject").unique())
    subjects = {row["Relation"]: row["Subject"] for row in pred2subj.to_dicts()}

    pred2obj = kg.group_by("Relation").agg(pl.col("Object").unique())
    objects = {row["Relation"]: row["Object"] for row in pred2obj.to_dicts()}

    # sets for fast membership
    kg_set = set(map(tuple, kg.iter_rows()))
    correct_set = set(map(tuple, correct_df.iter_rows()))
    noisy_set = set()

    print("Generating noisy triples...")
    for triple in tqdm(noisy_candidates.iter_rows()):
        new_triple = generate_noisy_triple(triple, subjects, objects, kg_set, noisy_set)
        if new_triple:
            noisy_set.add(new_triple)
        else:
            raise RuntimeError("Failed to generate a noisy triple")

    # convert to dict
    noisyKG, gt = {}, {}

    for i, t in enumerate(sorted(correct_set)):
        key = f"{i}_correct"
        noisyKG[key] = t
        gt[key] = 1

    for i, t in enumerate(sorted(noisy_set)):
        key = f"{i}_wrong"
        noisyKG[key] = t
        gt[key] = 0

    return noisyKG, gt


def generate_noisy_kg(raw_csv, output_root, acc_level):
    """
    Generate noisy KG for the given accuracy level.

    :param raw_csv: path to (filtered) KG
    :param output_root: path to output root
    :param acc_level: target accuracy level
    :return: None
    """

    print("Loading YAGO (filtered) KG...")
    kg = pl.read_csv(raw_csv)
    kg = kg.unique()
    print(f"YAGO (filtered) KG with {kg.height} triples loaded!")

    # prepare output
    out_dir = os.path.join(output_root, f"{str(acc_level)}/data")
    os.makedirs(out_dir, exist_ok=True)

    # set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    pl.set_random_seed(seed)

    # generate noisy KG
    noisyKG, noisyGT = corrupt_triples(kg, acc_level)

    # store files
    with open(os.path.join(out_dir, "kg.json"), "w") as f:
        json.dump(noisyKG, f)

    with open(os.path.join(out_dir, "gt.json"), "w") as f:
        json.dump(noisyGT, f)

    acc = sum(noisyGT.values()) / len(noisyGT)
    print(f"Stored YAGO corrupted KG with accuracy {acc:.2f}")
    print(f"Output folder: {out_dir}\n")


def main():
    filtered_csv = "./raw/filtered-facts.csv"
    if not os.path.exists(filtered_csv):
        filter_yago(args.inputYago, filtered_csv)

    generate_noisy_kg(raw_csv=filtered_csv, output_root=args.outputRoot, acc_level=args.accLevel)


if __name__ == "__main__":
    main()
