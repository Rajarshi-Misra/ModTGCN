import os
from openai import OpenAI
import pandas as pd
import numpy as np
import time
import logging

logging.basicConfig(filename='../artifacts/embedding_errors.log', level=logging.ERROR)

client = OpenAI(
    api_key="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"
)

datasets = ['20ng']
batch_size = 30

for dataset in datasets:
    splits = ['train', 'val', 'test']
    for split in splits:
        df = pd.read_csv(f"../inputs/20ng_cleaned/{split}.csv")
        texts = df['text'].tolist()
        print(dataset)
        print(split)
        print(len(texts))
        embeddings = []
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx:idx+batch_size]
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch
                )
                for emb_obj in response.data:
                    embeddings.append(emb_obj.embedding)
            except Exception as e:
                logging.error(f"{dataset} | {split} | batch_idx: {idx} | error: {e} | length: {len(texts[idx])}")
                time.sleep(3)  # Simple backoff
                continue
            if idx % 500 == 0:
                print(f"{dataset} - {split}: Processed {idx} texts")

        embeddings_array = np.array(embeddings)
        np.save(f"../embeddings/gpt/{dataset}/{split}.npy", embeddings_array)
