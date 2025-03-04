# Fake News Detection in Election-Related Misinformation
By Emma Corbett
## Background

The issue of widespread consumption of political and election-related misinformation, especially through social media, poses a significant threat to informed decision-making and democratic integrity. Studies indicate that 73% of U.S. adults encounter inaccurate election news, and 52% of them struggle to determine its authenticity. Platforms such as Facebook, which can hyper-personalize content, often create "information bubbles" that facilitate rapid misinformation spread, ultimately undermining trust in institutions and election outcomes.

This problem is particularly acute in the initial hours following major global events. During these times, traditional mainstream media and government sources can be slow to provide accurate details, giving disinformation sources a critical window to establish false narratives. According to experts at the European Values Center in Prague, responses to disinformation must be issued within two hours to effectively counter these narratives.

Despite bipartisan support for legislation targeting AI-driven electoral misinformation, the U.S. Congress has yet to pass substantive legislation. In the absence of such regulation, user-friendly, data-driven fact-checking tools are crucial for restoring public trust in the information ecosystem.

## Proposed Solution

Currently, successful fake news detection models primarily rely on supervised machine learning approaches, which depend on extensive hand-labeled data. However, creating such labeled datasets takes time, and the fast-paced nature of news on social media means that data can quickly become outdated. To address these limitations, we propose an **unsupervised learning approach** for detecting fake news, aiming to bypass the challenges faced by supervised models and provide a more adaptive, scalable solution for misinformation detection.

## Goals

1. **Enhance Rapid Response**: Develop tools that can identify misinformation within the critical two-hour window following a major event.
2. **Scalable Detection Models**: Leverage unsupervised learning techniques to adapt more quickly to evolving misinformation without the dependency on large, labeled datasets.
3. **Public Trust Restoration**: Build tools that empower citizens to fact-check information in real-time, supporting informed decisions and protecting democratic processes.

---

By moving away from traditional supervised models, our approach aims to create a more resilient detection framework capable of adapting to the fast-evolving nature of misinformation on social media.

# Research Question

## Primary Question
Can unsupervised learning approaches perform as well or better than supervised learning approaches in identifying text-based misinformation?

## Approach

There are two main approaches to identifying fake news:
1. **Content Analysis**: This involves examining the content of the news itself to identify false information.
2. **Propagation Pattern Analysis**: This approach focuses on how misinformation spreads across networks.

To explore both approaches, we will test two models:
- **Content-based Model**: Using clustering techniques to define contrasting viewpoints, as proposed by Jin et al. (2016), we will focus on analyzing content to detect misinformation.
- **Propagation Pattern Model**: Implementing a Graph Autoencoder with Masking and Contrastive Learning (GAMC) model based on the work of Yin et al. (2024). This model will analyze the propagation patterns of data by encoding and decoding the propagation graph.

## Evaluation Sub-questions

1. **Model Performance**: Which model achieves the best accuracy and consistency in predictions?
2. **Scalability**: How scalable is each model?
3. **Interpretability**: How interpretable are the model’s outputs?

## Analysis Sub-questions

1. **Reliability Comparison**: How do the model’s categorization results compare to established “reliability” scores such as PolitiFact’s “truth-o-meter”?
2. **Generalizability and Adaptability**: Do unsupervised models demonstrate greater adaptability to new misinformation trends?
3. **Bias Examination**: Depending on model interpretability, can we detect any political bias in the models?
4. **Unsupervised vs. Supervised Comparison**: What metrics allow for accurate comparison between unsupervised and supervised models?

## Console into Google Cloud VM
In order to console into the Google Cloud VM via your local machine, you will need to ***URL**[download and initialize the Google Cloud CLI](https://cloud.google.com/sdk/docs/install) on your local machine. 

## Using Datasets

Both datasets use PolitiFact as a source, and FakeNewsNet also draws from GossipCop. 

The FakeNews dataset can be accessed directly [here](https://github.com/KaiDMML/FakeNewsNet).

<<<<<<< Updated upstream
The LIAR dataset can be accessed via the [deeplake repo](https://github.com/activeloopai/deeplake). Loading the dataset will require importing deeplake into your files like so:
```bash
import deeplake
ds = deeplake.load('hub://activeloop/liar-train')
```
The same format will be used for test and validation sets. 
=======
2. **FakeNewsNet Dataset**: Comprising data from PolitiFact and GossipCop, this dataset categorizes stories as “real” or “fake.” We will remove these labels during training.

   - **URL**: [FakeNewsNet Dataset on GitHub](https://github.com/KaiDMML/FakeNewsNet)

---

This research will help assess whether unsupervised approaches provide a viable, scalable alternative to supervised models in the fight against misinformation.


# Login steps for connecting to cloud
gcloud auth login
gcloud compute ssh ec3745@fakenews --zone us-central1-a
then cd ../jq2347/ to get into the files under Julia's folder

