import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BertModel
from transformers import BertTokenizer
from torchtext.vocab import GloVe

'''
    This file contain methods to generate four different types of embeddings: 
    TF-IDF, Bag of words (Bow), Glove, and Bert.
    We use these methods to convert our dataset into each embedding type 
    to test which embedding performs the best with our models.
'''


# Goal: compare different embedding approaches and see performance on clustering
# TFIDF encoding
def get_tfidf_encoding(liar_df_train_topic):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(liar_df_train_topic['statement'])
    return tfidf_matrix


# Bow encoding
def get_bag_of_words_embedding(liar_df_train_topic):
    word_embed_bow = CountVectorizer()
    bow_matrix = word_embed_bow.fit_transform(liar_df_train_topic['statement'])
    return bow_matrix


# Glove encoding
def glove_sentence_embedding(sentence, max_length, embedding_dim, embeddings):
    words = sentence.split()
    num_words = min(len(words), max_length)
    embedding_sentence = np.zeros((max_length, embedding_dim))

    for i in range(num_words):
        word = words[i]
        if word in embeddings.stoi:
            embedding_sentence[i] = embeddings.vectors[embeddings.stoi[word]]

    return embedding_sentence.flatten()


def get_glove_embedding(liar_df_train_topic):
    embeddings = GloVe(name='6B', dim=100)

    # Set the maximum sentence length and embedding dimension
    max_length = 100
    embedding_dim = 100

    liar_df_train_topic['encode_glove'] = liar_df_train_topic['statement'].apply(
        lambda sentence: glove_sentence_embedding(sentence, max_length, embedding_dim, embeddings))
    X_glove = np.vstack(liar_df_train_topic['encode_glove'])
    return X_glove


# Bert embedding
def get_bert_embedding(liar_df_train_topic):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    liar_df_train_topic['cls_bert'] = liar_df_train_topic['statement'].apply(
        lambda sentence: get_cls_sentence(sentence, tokenizer, model))
    X_cls_bert = np.vstack(liar_df_train_topic['cls_bert'])
    return X_cls_bert


def get_cls_sentence(sentence, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True, max_length=512)])

    with torch.no_grad():
        outputs = model(input_ids)
        cls_embedding = outputs[0][:, 0, :]

    return cls_embedding.flatten()