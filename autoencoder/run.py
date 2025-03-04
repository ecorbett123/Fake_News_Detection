import argparse
import joblib
import os
from data_utils import load_data, create_embeddings
from sklearn.cluster import KMeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AE, VAE, or K-Means on a given dataset.")
    parser.add_argument('--subset', type=str, required=True, help="Path to the dataset subset file (tsv or csv)")
    parser.add_argument('--model', type=str, required=True, choices=['ae', 'vae', 'kmeans'],
                        help="Model type: ae, vae, or kmeans")
    parser.add_argument('--embedding_type', type=str, required=True,
                        choices=['bert', 'glove', 'word2vec'],
                        help="Embedding type: bert, glove, or word2vec")
    parser.add_argument('--bert_model_name', type=str, default='all-MiniLM-L6-v2',
                        help="BERT model name if embedding_type=bert")
    parser.add_argument('--glove_path', type=str, default=None,
                        help="Path to GloVe keyed vectors if embedding_type=glove")
    parser.add_argument('--word2vec_path', type=str, default=None,
                        help="Path to Word2Vec binary file if embedding_type=word2vec")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (for AE or VAE)")
    parser.add_argument('--latent_dim', type=int, default=64, help="Latent dimension size (for AE or VAE)")
    parser.add_argument('--n_clusters', type=int, default=8, help="Number of clusters for K-Means if model=kmeans")

    args = parser.parse_args()

    print("Arguments parsed, loading data now...")
    df = load_data(args.subset)
    print(f"{args.subset} loaded...")

    dataset_name = os.path.splitext(os.path.basename(args.subset))[0]

    if 'statement' not in df.columns and 'title' not in df.columns:
        if df.shape[1] == 14:
            df.columns = [
                "json_id",
                "label",
                "statement",
                "subject",
                "speaker",
                "job",
                "state_info",
                "party",
                "barely_true_count",
                "false_count",
                "half_true_count",
                "mostly_true_count",
                "pants_on_fire_count",
                "context"
            ]
            print("Assigned LIAR columns with 'json_id' for this custom dataset.")
        else:
            raise ValueError("No suitable text column found and dataset is not recognized.")

    if 'statement' in df.columns:
        text_column = 'statement'
    elif 'title' in df.columns:
        text_column = 'title'
    else:
        raise ValueError("No suitable text column found (neither 'statement' nor 'title').")

    texts = df[text_column].astype(str).tolist()

    print(f"Creating {args.embedding_type} embeddings...")
    embeddings = create_embeddings(texts,
                                   embedding_type=args.embedding_type,
                                   bert_model_name=args.bert_model_name,
                                   glove_path=args.glove_path,
                                   word2vec_path=args.word2vec_path)
    print(f"{args.embedding_type} embeddings created...")

    os.makedirs('trained_models', exist_ok=True)

    if args.model == 'ae':
        from train import train_autoencoder
        train_autoencoder(embeddings, epochs=args.epochs, latent_dim=args.latent_dim)
        model_filename = f"ae_model_{dataset_name}_{args.embedding_type}.pth"
        if os.path.exists('autoencoder_model.pth'):
            new_path = os.path.join('trained_models', model_filename)
            os.rename('autoencoder_model.pth', new_path)
            print(f"Autoencoder model saved as {new_path}")

    elif args.model == 'vae':
        from train import train_vae
        train_vae(embeddings, epochs=args.epochs, latent_dim=args.latent_dim)
        model_filename = f"vae_model_{dataset_name}_{args.embedding_type}.pth"
        if os.path.exists('vae_model.pth'):
            new_path = os.path.join('trained_models', model_filename)
            os.rename('vae_model.pth', new_path)
            print(f"VAE model saved as {new_path}")

    elif args.model == 'kmeans':
        print(f"Running K-Means with {args.n_clusters} clusters...")
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        print("K-Means completed.")
        model_filename = f"kmeans_model_{dataset_name}_{args.embedding_type}.pkl"
        new_path = os.path.join('trained_models', model_filename)
        joblib.dump(kmeans, new_path)
        print(f"K-Means model saved as {new_path}")
        print("Cluster assignments for each sample:", labels)
