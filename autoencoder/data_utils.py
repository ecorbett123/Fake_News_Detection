import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors

def load_data(file_path):
    """
    Load a TSV or CSV file and return a pandas DataFrame.
    """
    _, ext = os.path.splitext(file_path)
    sep = '\t' if ext.lower() == '.tsv' else ','
    df = pd.read_csv(file_path, sep=sep)
    return df

def create_embeddings(texts, embedding_type='bert',
                      bert_model_name='all-MiniLM-L6-v2',
                      glove_path=None,
                      word2vec_path=None):
    """
    Create embeddings for a list of texts using the specified embedding_type.
    
    embedding_type: 'bert', 'glove', or 'word2vec'
    bert_model_name: if using bert embeddings
    glove_path: path to GloVe vectors (in word2vec format) if embedding_type='glove'
    word2vec_path: path to word2vec binary if embedding_type='word2vec'
    """
    if embedding_type == 'bert':
        model = SentenceTransformer(bert_model_name)
        embeddings = model.encode(texts)

    elif embedding_type == 'glove':
        if glove_path is None:
            raise ValueError("glove_path must be provided for GloVe embeddings.")
        glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False)
        embeddings = []
        for text in texts:
            tokens = text.split()
            token_vecs = [glove_model[word] for word in tokens if word in glove_model]
            embeddings.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(glove_model.vector_size))
        embeddings = np.array(embeddings)

    elif embedding_type == 'word2vec':
        if word2vec_path is None:
            raise ValueError("word2vec_path must be provided for Word2Vec embeddings.")
        w2v_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        embeddings = []
        for text in texts:
            tokens = text.split()
            token_vecs = [w2v_model[word] for word in tokens if word in w2v_model]
            embeddings.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(w2v_model.vector_size))
        embeddings = np.array(embeddings)

    else:
        raise ValueError(f"Unsupported embedding_type: {embedding_type}")

    return embeddings
