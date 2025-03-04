import os
import torch
import joblib
import numpy as np

from atproto import Client
from data_utils import create_embeddings
from models import Autoencoder, VAE

HANDLE = 'fakenewsbot.bsky.social'
APP_PASSWORD = 'fwX$147Aeg5!t3qhoXR@'

client = Client()
client.login(HANDLE, APP_PASSWORD)
print("Logged into Bsky!")

TRENDING_NEWS_FEED_URI = "at://did:plc:kkf4naxqmweop7dv4l2iqqf5/app.bsky.feed.generator/news-2-0"
BREAKING_NEWS_FEED_URI = "at://did:plc:kkf4naxqmweop7dv4l2iqqf5/app.bsky.feed.generator/news"

trending_response = client.app.bsky.feed.get_feed(params={"feed": TRENDING_NEWS_FEED_URI})
breaking_response = client.app.bsky.feed.get_feed(params={"feed": BREAKING_NEWS_FEED_URI})


def load_models_from_dir(directory='trained_models'):
    models_info = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        name, ext = os.path.splitext(filename)
        parts = name.split('_')
        if len(parts) < 5:
            continue

        model_type = parts[0]

        embedding_type = parts[-1]
        dataset_parts = parts[2:-1]
        dataset_name = "_".join(dataset_parts)

        if model_type in ['ae', 'vae'] and ext == '.pth':
            input_dim = 384
            latent_dim = 64
            if model_type == 'ae':
                m = Autoencoder(input_dim, latent_dim)
            else:
                m = VAE(input_dim, latent_dim)
            m.load_state_dict(torch.load(filepath, weights_only=True))
            m.eval()
            models_info.append({
                'model_type': model_type,
                'dataset_name': dataset_name,
                'embedding_type': embedding_type,
                'model': m
            })

        elif model_type == 'kmeans' and ext == '.pkl':

            kmeans = joblib.load(filepath)
            models_info.append({
                'model_type': model_type,
                'dataset_name': dataset_name,
                'embedding_type': embedding_type,
                'model': kmeans
            })

    return models_info

models_info = load_models_from_dir()

ae_threshold = 0.02
vae_threshold = 0.02
kmeans_threshold = 1.5 

def compute_reconstruction_error(model, emb, vae=False):
    emb_t = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        if vae:
            x_recon, mu, logvar = model(emb_t)
            return ((x_recon - emb_t)**2).mean().item()
        else:
            x_recon = model(emb_t)
            return ((x_recon - emb_t)**2).mean().item()

def compute_kmeans_distance(kmeans_model, emb):
    cluster = kmeans_model.predict(emb)[0]
    center = kmeans_model.cluster_centers_[cluster]
    dist = np.linalg.norm(emb[0] - center)
    return dist

def score_post(text, embedding_type='bert', bert_model_name='all-MiniLM-L6-v2'):
    emb = create_embeddings([text], embedding_type=embedding_type, bert_model_name=bert_model_name)
    suspicious_models = []
    for info in models_info:
        model_type = info['model_type']
        dataset_name = info['dataset_name']
        embed_type = info['embedding_type']
        model = info['model']

        if model_type in ['ae', 'vae']:
            vae_flag = (model_type == 'vae')
            err = compute_reconstruction_error(model, emb, vae=vae_flag)
            if model_type == 'ae' and err > ae_threshold:
                suspicious_models.append((model_type, dataset_name, embed_type))
            elif model_type == 'vae' and err > vae_threshold:
                suspicious_models.append((model_type, dataset_name, embed_type))

        elif model_type == 'kmeans':
            dist = compute_kmeans_distance(model, emb)
            if dist > kmeans_threshold:
                suspicious_models.append((model_type, dataset_name, embed_type))

    return suspicious_models

def format_suspicious_message(suspicious_models, text):
    def model_description(mtype):
        if mtype == 'ae':
            return "an autoencoder"
        elif mtype == 'vae':
            return "a variational autoencoder"
        elif mtype == 'kmeans':
            return "a K-Means model"
        return "a model"

    models_str_list = []
    for (mtype, dset, etype) in suspicious_models:
        desc = f"{model_description(mtype)} trained on the {dset} dataset using {etype} embeddings"
        models_str_list.append(desc)

    if len(models_str_list) == 1:
        model_list_str = models_str_list[0]
    elif len(models_str_list) == 2:
        model_list_str = " and ".join(models_str_list)
    else:
        model_list_str = ", ".join(models_str_list[:-1]) + " and " + models_str_list[-1]

    return f"{model_list_str.capitalize()} has identified this post as potentially suspicious: {text[:100]}..."

def process_post(post_entry):
    text = getattr(post_entry.post.record, 'text', '') or ''
    author = post_entry.post.author.handle
    uri = post_entry.post.uri

    suspicious_models = score_post(text)
    print(f"Author: {author}\nText: {text}\nURI: {uri}")

    if suspicious_models:
        msg = format_suspicious_message(suspicious_models, text)
        print("This post may be suspicious.")
        client.app.bsky.feed.post(text=msg)
    else:
        print("This post seems normal.")

    print("---")

for post_entry in trending_response.feed:
    process_post(post_entry)

for post_entry in breaking_response.feed:
    process_post(post_entry)
