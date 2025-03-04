import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from textblob import TextBlob
from liar_preprocess import preprocess_text
from create_embeddings import get_bert_embedding, get_tfidf_encoding, get_glove_embedding, get_bag_of_words_embedding
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

'''
    This file explores performance of KMeans clustering on the LIAR dataset 
    when data points all belong to the same topic. It first explores the optimal k value,
    then measures the performance for the clusters by using entropy measurements. 
    It also produces visualizations of label distribution within each cluster.  
'''

warnings.filterwarnings(action='ignore')

liar_df_train = pd.read_csv('/Users/emmacorbett/PycharmProjects/coms6998mlf/liar_dataset/train.tsv', sep='\t', header=None)
liar_df_train.columns = ["id", "label", "statement", "subject", "Speaker", "Job_Title", "State_Info", "Party_Affiliation", "Barely_True_Counts", "False_Counts", "Half_True_Counts", "Mostly_True_Counts", "Pants_On_Fire_Counts", "Context"]
liar_df_train.drop(columns=['id', 'Speaker', 'Job_Title', 'State_Info', 'Party_Affiliation', 'Barely_True_Counts', 'False_Counts', 'Half_True_Counts', 'Mostly_True_Counts', 'Pants_On_Fire_Counts', 'Context'], inplace=True)
liar_df_train['statement'] = [preprocess_text(text) for text in liar_df_train['statement']]


def get_all_subsets():
    # find all unique subjects
    subjects = set()
    for subject in liar_df_train['subject']:
        subject_list = str(subject).split(',')
        subjects.update(subject_list)
    return subjects


def create_subject_comment_map():
    # sort comments based on subject
    subject_comment_map = {}
    subject_comment_count_map = {}
    for index, row in liar_df_train.iterrows():
        subject_list = str(row['subject']).split(',')
        for subject in subject_list:
            if subject not in subject_comment_map:
                subject_comment_map[subject] = [row['statement']]
            else:
                subject_comment_map[subject].append(row['statement'])

    # See distribution of comments to subject
    for k, v in subject_comment_map.items():
        subject_comment_count_map[k] = len(v)
    return subject_comment_map, subject_comment_count_map


subject_comment_map, subject_comment_count_map = create_subject_comment_map()

# get top topic and top 10 topics
def get_top_10_topics():
    dict(sorted(subject_comment_count_map.items(), key=lambda item: item[1], reverse=True))
    top_10_subjects = []
    count = 0
    for k, v in subject_comment_count_map.items():
        top_10_subjects.append(k)
        count += 1
        if count == 10:
            break
    return top_10_subjects


top_10_subjects = get_top_10_topics()
largest_subject = top_10_subjects[0]

# append sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'


# add sentiment analysis to data set
liar_df_train['sentiment'] = liar_df_train['statement'].apply(get_sentiment)


def get_topic_dataframe(topic):
    liar_df_train_topic = pd.DataFrame()
    for index, row in liar_df_train.iterrows():
        subject_list = str(row['subject']).split(',')
        if str(topic) in subject_list:
            liar_df_train_topic = liar_df_train_topic._append(row, ignore_index=True)
    return liar_df_train_topic


# entropy function to evaluate clusters
# lower entropy means more homogeneous cluster
def entropy(label_distribution_val):
    entropy_val = 0
    total = 0
    for ke, ve in label_distribution_val.items():
        total += int(ve)

    for ke, ve in label_distribution_val.items():
        p_i = int(ve)/total
        entropy_val += (p_i * math.log2(p_i))

    return entropy_val*-1


# calculate the entropy and return in proportion of cluster size to total number of statements
def entropy_cluster_size(label_distribution_val, total_num_statements_val):
    entropy_val = 0
    cluster_size = 0
    for ky, vy in label_distribution_val.items():
        cluster_size += int(vy)

    for ky, vy in label_distribution_val.items():
        p_i = int(vy)/cluster_size
        entropy_val += (p_i * math.log2(p_i))

    return entropy_val*-1*cluster_size/total_num_statements_val


# test top 10 subejcts cluster performance... uncomment embedding of choice; Bow & tfidf perform the best
for subject in top_10_subjects:
    liar_df_train_topic = get_topic_dataframe(subject)
    #X_cls_bert = get_bert_embedding(liar_df_train_topic)
    #X_tfidf = get_tfidf_encoding(liar_df_train_topic)
    X_bow = get_bag_of_words_embedding(liar_df_train_topic)
    #X_glove = get_glove_embedding(liar_df_train_topic)

    # PCA dimension reduction
    pca = PCA(n_components=2, random_state=42)
    X = pca.fit_transform(X_bow)

    # train model on top topic
    print(f"Subject: {subject}")
    # use wcss to plot elbow point graph to help pick optimal value of k
    # result, k = 6 is optimal
    wcss = []
    for num_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        liar_df_train_topic['cluster'] = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)

        # Analyze clusters
        for cluster in range(num_clusters):
            cluster_data = liar_df_train_topic[liar_df_train_topic['cluster'] == cluster]

        # see distribution of sentiment (pos/neut/neg) for each cluster
        for cluster in range(num_clusters):
            cluster_sentiments = liar_df_train_topic[liar_df_train_topic['cluster'] == cluster]['sentiment']
            sentiment_distribution = cluster_sentiments.value_counts()
            print(f"Cluster {cluster} sentiment distribution:")
            print(sentiment_distribution)

        # get entropy measurements for each clustering model
        total_entropy = 0
        total_entropy_weighted = 0
        vals = []
        for cluster in range(num_clusters):
            total_num_statements = len(liar_df_train_topic)
            cluster_labels = liar_df_train_topic[liar_df_train_topic['cluster'] == cluster]['label']
            label_distribution = cluster_labels.value_counts()
            vals.append(label_distribution)
            total_entropy_weighted += entropy_cluster_size(label_distribution, total_num_statements)
            total_entropy += entropy(label_distribution)

        kmeans_labels = kmeans.labels_
        plt.figure(figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='rainbow')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='centroid')
        plt.title('Kmeans Clustering')
        plt.show()
        silhouette_score_kmeans = silhouette_score(X, kmeans_labels)
        print(f"Num Clusters: {num_clusters}, Silhouette Score: {silhouette_score_kmeans}")

        # Plot distribution of labels per cluster
        categories = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        x = np.arange(len(categories))  # Positions for the first set of bars
        num = 1
        bar_width = 0.15
        num_bar_width = -3
        for val in vals:
            new_val = []
            for cat in categories:
                if cat in val:
                    new_val.append(val[cat])
                else:
                    new_val.append(0)
            # Plot
            plt.bar(x + (num_bar_width*bar_width/2), new_val, width=0.15, label=f'Cluster {num}')  # First bar
            num += 1
            num_bar_width += 2

        # Add labels and legend
        plt.xticks(x, categories)
        plt.title(f"K={num_clusters} Cluster Label Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        print(f"Num Clusters: {num_clusters}, Average Entropy Per Cluster: {total_entropy/num_clusters}, Average Weighted Entropy: {total_entropy_weighted / num_clusters}")

    # find optimal k value using elbow point graph
    sns.set()
    plt.plot(range(2, 11), wcss)

    plt.title('The Elbow Point Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')

    plt.show()

# optimal models analysis for largest topic
# liar_df_train_topic = get_topic_dataframe(largest_subject)
# #X_cls_bert = get_bert_embedding(liar_df_train_topic)
# #X_tfidf = get_tfidf_encoding(liar_df_train_topic)
# X_bow = get_bag_of_words_embedding(liar_df_train_topic)
# #X_glove = get_glove_embedding(liar_df_train_topic)
#
# # PCA dimension reduction
# pca = PCA(n_components=2, random_state=42)
# X = pca.fit_transform(X_bow)
#
# # train model on top topic
# print(f"Subject: {largest_subject}")
# kmeans = KMeans(n_clusters=6, random_state=42)
# liar_df_train_topic['cluster'] = kmeans.fit_predict(X)
# #liar_df_train_topic['cluster'] = kmeans.fit(X)
#
# total_entropy = 0
# total_entropy_weighted = 0
# vals = []
# for cluster in range(6):
#     total_num_statements = len(liar_df_train_topic)
#     cluster_labels = liar_df_train_topic[liar_df_train_topic['cluster'] == cluster]['label']
#     label_distribution = cluster_labels.value_counts()
#     vals.append(label_distribution)
#     total_entropy_weighted += entropy_cluster_size(label_distribution, total_num_statements)
#     total_entropy += entropy(label_distribution)
#
# kmeans_labels = kmeans.labels_
# silhouette_score_kmeans = silhouette_score(X, kmeans_labels)
# print(f"Num Clusters: 6, Silhouette Score: {silhouette_score_kmeans}")
# print(f"Num Clusters: 6, Average Entropy Per Cluster: {total_entropy/6}, Average Weighted Entropy: {total_entropy_weighted / 6}")

# Results from above largest topic performance analysis: Bag of words performs the best
# glove
# Num Clusters: 6, Silhouette Score: 0.35786499511104397
# Num Clusters: 6, Average Entropy Per Cluster: 2.36900475122478, Average Weighted Entropy: 0.4068388948097495

# tfidf
# Num Clusters: 6, Silhouette Score: 0.4388068356387779
# Num Clusters: 6, Average Entropy Per Cluster: 1.9980184784086286, Average Weighted Entropy: 0.4064072496532229

# bert
# Num Clusters: 6, Silhouette Score: 0.341304212808609
# Num Clusters: 6, Average Entropy Per Cluster: 2.3999556924710492, Average Weighted Entropy: 0.40254100217920746

# bow
# Num Clusters: 6, Silhouette Score: 0.5020987624661493
# Num Clusters: 6, Average Entropy Per Cluster: 2.4329580889773936, Average Weighted Entropy: 0.40743915794007285


# Code to determine optimal k-value for entire data set, not topic specific... optimal value is k=6 as well
# X_cls_bert = get_bert_embedding(liar_df_train)
#
# # PCA dimension reduction
# pca = PCA(n_components=2, random_state=42)
# X = pca.fit_transform(X_cls_bert)
#
# wcss = []
# for num_clusters in range(1, 11):
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     liar_df_train['cluster'] = kmeans.fit_predict(X_cls_bert)
#     wcss.append(kmeans.inertia_)
#
#     # Analyze clusters
#     for cluster in range(num_clusters):
#         cluster_data = liar_df_train[liar_df_train['cluster'] == cluster]
#
#     total_entropy = 0
#     total_entropy_weighted = 0
#     for cluster in range(num_clusters):
#         total_num_statements = len(liar_df_train)
#         cluster_labels = liar_df_train[liar_df_train['cluster'] == cluster]['label']
#         label_distribution = cluster_labels.value_counts()
#         total_entropy_weighted += entropy_cluster_size(label_distribution, total_num_statements)
#         total_entropy += entropy(label_distribution)
#
#     print(f"Num Clusters: {num_clusters}, Average Entropy Per Cluster: {total_entropy/num_clusters}, Average Weighted Entropy: {total_entropy_weighted / num_clusters}")
#
# sns.set()
# plt.plot(range(1, 11), wcss)
#
# plt.title('The Elbow Point Graph')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
#
# plt.show()
