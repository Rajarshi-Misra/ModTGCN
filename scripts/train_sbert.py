import os
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from sentence_transformers import models
import random
import numpy as np
import argparse
import torch
import pandas as pd
from collections import defaultdict

def make_positive_pairs_from_labels(df, num_pairs_per_label=1000):
    groups = defaultdict(list)
    for _, row in df.iterrows():
        lbl = row['label']
        groups.setdefault(lbl, []).append(str(row['text']))

    examples = []
    for lbl, texts in groups.items():
        if len(texts) < 2:
            continue  # Need at least 2 for anchor/positive
        for _ in range(min(num_pairs_per_label, len(texts) * 2)):
            a, b = random.sample(texts, 2)
            examples.append(InputExample(texts=[a, b]))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name used for paths")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2", help="pretrained SBERT model")
    parser.add_argument("--output_dir", type=str, default="../models/sbert", help="base output dir")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_csv = f"../inputs/{args.dataset}/train.csv"
    val_csv = f"../inputs/{args.dataset}/val.csv"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    train_examples = make_positive_pairs_from_labels(df_train, num_pairs_per_label=args.num_pairs)
    val_examples = make_positive_pairs_from_labels(df_val, num_pairs_per_label=args.num_pairs)

    model = SentenceTransformer(args.model_name, device=args.device)

    train_dataset = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, collate_fn=model.smart_batching_collate)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if len(val_examples) > 0:
        # create pairs with label 1; and create some negative pairs with label 0
        pairs = []
        for ex in val_examples[:1000]:
            pairs.append((ex.texts[0], ex.texts[1], 1.0))
        # create some negatives by pairing random texts across classes
        texts_by_label = {}
        for _, row in df_val.iterrows():
            texts_by_label.setdefault(row['label'], []).append(str(row['text']))
        labels = list(texts_by_label.keys())
        # make negatives
        for _ in range(min(500, len(pairs))):
            a_label, b_label = random.sample(labels, 2)
            a = random.choice(texts_by_label[a_label])
            b = random.choice(texts_by_label[b_label])
            pairs.append((a, b, 0.0))
        sentences1 = [p[0] for p in pairs]
        sentences2 = [p[1] for p in pairs]
        scores = [p[2] for p in pairs]
        evaluator = evaluation.BinaryClassificationEvaluator(sentences1, sentences2, scores, show_progress_bar=False)

    # training
    output_path = os.path.join(args.output_dir, args.dataset, "checkpoint")
    os.makedirs(output_path, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataset, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=max(100, len(train_dataset) * args.epochs // 10),
        output_path=output_path,
        optimizer_params={'lr': args.lr},
        checkpoint_path=os.path.join(args.output_dir, args.dataset, "checkpoint-ckpt"),
        save_best_model=True
    )

    print("Training complete. Model saved to:", output_path)

if __name__ == "__main__":
    main()