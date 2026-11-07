"""Step 2: encode the abstracts with config.EMBEDDING_MODEL into bertopic/preprocessed_data/embeddings.npz.

Only the papers of alldata.csv whose paper_id has no saved embedding yet are encoded. If the encoder settings
(model, dimension, prompt; config.encoder()) differ from those the saved embeddings were made with, every paper is
encoded again, since embeddings of different encoders can't be mixed.
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import config


def embed(docs):
    # float16 halves the memory of the 4B model (bfloat16 is slow on Apple GPUs); the embeddings are saved as float16
    model = SentenceTransformer(config.EMBEDDING_MODEL, truncate_dim=config.EMBEDDING_DIM,
                                model_kwargs={'dtype': torch.float16})
    embeddings = model.encode(docs, prompt=config.EMBEDDING_PROMPT, batch_size=8, show_progress_bar=True,
                              convert_to_numpy=True).astype(np.float32)
    # Normalize after the truncation
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def main():
    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)
    if config.EMBEDDINGS.exists() and config.saved_encoder() == config.encoder():
        ids, embeddings = config.load_embeddings()
    else:
        if config.EMBEDDINGS.exists():
            print(f"Saved embeddings were made with {config.saved_encoder()}, not {config.encoder()}: encoding all")
        ids, embeddings = np.array([], dtype=str), np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)
    new = data[~data['paper_id'].isin(ids)]
    print(f"Encoding {len(new)} new abstracts ({len(ids)} already done)")
    if not len(new):
        return

    config.save_embeddings(np.concatenate([ids, new['paper_id']]),
                           np.concatenate([embeddings, embed(new['Abstract'].tolist())]))


if __name__ == '__main__':
    main()
