import pandas as pd
from sklearn.cluster import KMeans
from create_embeddings import get_bert_embedding, get_tfidf_encoding, get_glove_embedding, get_bag_of_words_embedding
from sklearn.decomposition import PCA
from liar_preprocess import preprocess_text
from liar_model_cluster_by_topic import entropy
import joblib

'''
    This file explores performance of KMeans clustering on the entire LIAR dataset. 
    Optimal k values were explored in the liar_model_cluster_by_topic file, and then
    used here to measure the performance of the clusters by using entropy measurements.
    The models are then saved for later use, one for each type of word embedding (tfidf, glove, bert, bow). 
'''


# Used to test kmeans when data is balanced according to outcome variable... did not make much of a difference
def create_balanced_subset(df, column):
    """Creates a subset of the DataFrame with an equal number of values for the specified column."""

    # Find the minimum count of each unique value in the column
    min_count = df[column].value_counts().min()

    # Create a new DataFrame for the subset
    subset_df = pd.DataFrame()

    # Iterate through unique values in the column
    for value in df[column].unique():
        # Sample the DataFrame for the current value
        subset_df = pd.concat([subset_df, df[df[column] == value].sample(min_count)])

    return subset_df

# run model on entire dataset
liar_df_train = pd.read_csv('/Users/emmacorbett/PycharmProjects/coms6998mlf/liar_dataset/train.tsv', sep='\t', header=None)
liar_df_train.columns = ["id", "label", "statement", "subject", "Speaker", "Job_Title", "State_Info", "Party_Affiliation", "Barely_True_Counts", "False_Counts", "Half_True_Counts", "Mostly_True_Counts", "Pants_On_Fire_Counts", "Context"]
liar_df_train.drop(columns=['id', 'Speaker', 'Job_Title', 'State_Info', 'Party_Affiliation', 'Barely_True_Counts', 'False_Counts', 'Half_True_Counts', 'Mostly_True_Counts', 'Pants_On_Fire_Counts', 'Context'], inplace=True)
liar_df_train['statement'] = [preprocess_text(text) for text in liar_df_train['statement']]

# Bert embedding
X_cls_bert = get_bert_embedding(liar_df_train)
# PCA dimension reduction
pca_bert = PCA(n_components=2, random_state=42)
X_bert = pca_bert.fit_transform(X_cls_bert)
kmeans_bert = KMeans(n_clusters=6, random_state=42)
liar_df_train['cluster'] = kmeans_bert.fit_predict(X_bert)

for cluster in range(6):
    cluster_data = liar_df_train[liar_df_train['cluster'] == cluster]

total_entropy = 0
entropy_vals = []
for cluster in range(6):
    cluster_labels = liar_df_train[liar_df_train['cluster'] == cluster]['label']
    label_distribution = cluster_labels.value_counts()
    cluster_entropy = entropy(label_distribution)
    total_entropy += cluster_entropy
    entropy_vals.append(cluster_entropy)

print(f"BERT- Average Entropy Per Cluster: {total_entropy/6}, Entropy Vals: {entropy_vals}")
joblib.dump(kmeans_bert, 'liar_bert_full_kmeans_model.pkl')

# Glove
X_cls_glove = get_glove_embedding(liar_df_train)

# PCA dimension reduction
pca_glove = PCA(n_components=2, random_state=42)
X_cls_glove = pca_glove.fit_transform(X_cls_glove)
kmeans_glove = KMeans(n_clusters=6, random_state=42)
liar_df_train['cluster'] = kmeans_glove.fit_predict(X_cls_glove)

for cluster in range(6):
    cluster_data = liar_df_train[liar_df_train['cluster'] == cluster]

total_entropy = 0
entropy_vals = []
for cluster in range(6):
    cluster_labels = liar_df_train[liar_df_train['cluster'] == cluster]['label']
    label_distribution = cluster_labels.value_counts()
    cluster_entropy = entropy(label_distribution)
    total_entropy += cluster_entropy
    entropy_vals.append(cluster_entropy)

print(f"Glove- Average Entropy Per Cluster: {total_entropy/6}, Entropy Vals: {entropy_vals}")
joblib.dump(kmeans_glove, 'liar_glove_full_kmeans_model.pkl')

# TFIDF
X_cls_tfidf = get_tfidf_encoding(liar_df_train)

# PCA dimension reduction
pca_tfidf = PCA(n_components=2, random_state=42)
X_cls_tfidf = pca_tfidf.fit_transform(X_cls_tfidf)
kmeans_tfidf = KMeans(n_clusters=6, random_state=42)
liar_df_train['cluster'] = kmeans_tfidf.fit_predict(X_cls_tfidf)

for cluster in range(6):
    cluster_data = liar_df_train[liar_df_train['cluster'] == cluster]

total_entropy = 0
entropy_vals = []
for cluster in range(6):
    cluster_labels = liar_df_train[liar_df_train['cluster'] == cluster]['label']
    label_distribution = cluster_labels.value_counts()
    cluster_entropy = entropy(label_distribution)
    total_entropy += cluster_entropy
    entropy_vals.append(cluster_entropy)

print(f"Tfidf- Average Entropy Per Cluster: {total_entropy/6}, Entropy Vals: {entropy_vals}")
joblib.dump(kmeans_tfidf, 'liar_tfidf_full_kmeans_model.pkl')

# Bow
X_cls_bow = get_bag_of_words_embedding(liar_df_train)

# PCA dimension reduction
pca_bow = PCA(n_components=2, random_state=42)
X_cls_bow = pca_bow.fit_transform(X_cls_bow)
kmeans_bow = KMeans(n_clusters=6, random_state=42)
liar_df_train['cluster'] = kmeans_bow.fit_predict(X_cls_bow)

for cluster in range(6):
    cluster_data = liar_df_train[liar_df_train['cluster'] == cluster]

total_entropy = 0
entropy_vals = []
for cluster in range(6):
    cluster_labels = liar_df_train[liar_df_train['cluster'] == cluster]['label']
    label_distribution = cluster_labels.value_counts()
    cluster_entropy = entropy(label_distribution)
    total_entropy += cluster_entropy
    entropy_vals.append(cluster_entropy)

print(f"Bow- Average Entropy Per Cluster: {total_entropy/6}, Entropy Vals: {entropy_vals}")
joblib.dump(kmeans_bow, 'liar_bow_full_kmeans_model.pkl')

# Load model
kmeans_tfidf = joblib.load('liar_tfidf_full_kmeans_model.pkl')

# Get test data, label according to model, see how label corrsponds to cluster label
liar_df_test = pd.read_csv('/Users/emmacorbett/PycharmProjects/coms6998mlf/liar_dataset/test.tsv', sep='\t', header=None)
liar_df_test.columns = ["id", "label", "statement", "subject", "Speaker", "Job_Title", "State_Info", "Party_Affiliation", "Barely_True_Counts", "False_Counts", "Half_True_Counts", "Mostly_True_Counts", "Pants_On_Fire_Counts", "Context"]
liar_df_test.drop(columns=['id', 'Speaker', 'Job_Title', 'State_Info', 'Party_Affiliation', 'Barely_True_Counts', 'False_Counts', 'Half_True_Counts', 'Mostly_True_Counts', 'Pants_On_Fire_Counts', 'Context'], inplace=True)
liar_df_test['statement'] = [preprocess_text(text) for text in liar_df_test['statement']]

# get encoding
X_test_tfidf = get_tfidf_encoding(liar_df_test)
pca = PCA(n_components=2, random_state=42)
X_test_tfidf = pca.fit_transform(X_test_tfidf)
y_test_tfidf = liar_df_test['label']

y_pred_tfidf = kmeans_tfidf.predict(X_test_tfidf)
liar_df_test['predictions'] = y_pred_tfidf
counts = liar_df_test.groupby('label')['predictions'].value_counts()
print(counts)

# ffn analysis - note, need to change create embeddings file column to title instead of statement
# ffn_df_train = pd.read_csv("/Users/emmacorbett/PycharmProjects/coms6998mlf/models/test/datasets/FakeNewsNet.csv")
# ffn_df_train['title'] = [preprocess_text(text) for text in ffn_df_train['title']]
# X_train_tfidf = get_tfidf_encoding(ffn_df_train)
# pca = PCA(n_components=2, random_state=42)
# X_train_tfidf = pca.fit_transform(X_train_tfidf)
# y_train_tfidf = ffn_df_train['real']
#
# kmeans = KMeans(n_clusters=2, random_state=42)
# ffn_df_train['predictions'] = kmeans.fit_predict(X_train_tfidf)
# counts = ffn_df_train.groupby('real')['predictions'].value_counts()
# joblib.dump(kmeans, 'ffn_tfidf_kmeans_model.pkl')
# joblib.dump(pca, 'ffn_tfidf_pca_model.pkl')
# print(counts)
# print(f"Accuracy: {(919+17163)/(4836+919+17163+278)}")
