from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import hdbscan
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import itertools
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, encoded_inputs):
        self.encoded_inputs = encoded_inputs

    def __len__(self):
        return self.encoded_inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encoded_inputs.items()}


class TopicModeling:
    def __init__(self, text, model_name, device='cuda', self_sim_threshold=0.5, batch_size=32):
        self.corpus = text
        self.self_sim_threshold = self_sim_threshold
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encoding(self):
        print("Tokenizing text with model:")
        encoded_input = self.tokenizer(self.corpus, padding=True, truncation=True, return_tensors='pt')
        print("Finished Tokenizing:")

        dataset = MyDataset(encoded_input)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        print("Encoding text:")
        total_output = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.encoder.encode(batch['input_ids'], batch['attention_mask'])
                total_output.append(output)

        total_output = torch.cat(total_output, dim=0)
        print("Finished encoding")

        embedding = total_output
        globals()['embedding'] = embedding.cpu().numpy()
        return embedding.cpu().numpy()

    def self_similarity(self, last_hidden_state, token_list, inference_list):
        ss_score = {}
        temp = last_hidden_state
        token_embeddings_dict = {token: [] for token in token_list}

        for token in tqdm(token_list, desc='Token Progress'):
            token_embeddings = []
            for (sentence_index, token_index) in inference_list[token]:
                token_embedding = np.array(temp[sentence_index][token_index])
                token_embeddings.append(token_embedding)
            token_embeddings = np.array(token_embeddings)
            token_embeddings = token_embeddings.reshape(-1, 1)
            sim_matrix = cosine_similarity(token_embeddings, token_embeddings)
            if len(sim_matrix) != 1:
                self_similarity = round(
                    (np.sum(sim_matrix) - len(sim_matrix)) / (len(sim_matrix) * (len(sim_matrix) - 1)), 3)
                ss_score[token] = self_similarity

        return ss_score

    def build_candidates(self):
        print("Tokenizing text with model:")
        encoded_input = self.tokenizer(self.corpus, padding=True, truncation=True, return_tensors='pt')
        print("Finished Tokenizing:")
        tokenized = encoded_input['input_ids'].tolist()
        token_list = list(itertools.chain.from_iterable(tokenized))
        counter = Counter(token_list)

        candidate_vocab = [index for (index, count) in counter.most_common() if
                           count >= 5]  # filter out less frequent tokens
        last_hidden_state = self.encoding()

        inference_list = {}
        for n in set(candidate_vocab):
            position_list = []
            for sen_index, sen in enumerate(tokenized):
                if n in sen:
                    token_index = sen.index(n)
                    position_list.append((sen_index, token_index))
            inference_list[n] = position_list

        ss_score = self.self_similarity(last_hidden_state, candidate_vocab, inference_list)
        filtered_candidate_vocab = [token for token, self_sim in ss_score.items() if
                                    self_sim >= self.self_sim_threshold]

        return filtered_candidate_vocab

    def DR(self, embedding, dimension=5):
        reducer = umap.UMAP(random_state=42, n_components=dimension)
        embedding = reducer.fit_transform(embedding)
        return embedding

    def clustering(self, reduced_embedding, min_cluster_size=2):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(reduced_embedding)
        outlier_scores = clusterer.outlier_scores_
        return cluster_labels, outlier_scores

    def agglomerative_clustering(self, embedding, n_clusters=5):
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embedding)
        cluster_labels = clustering.labels_
        return cluster_labels, _

    def centroid(self, reduced_embedding, filtered_candidate_vocab, cluster_labels):
        text = self.corpus
        encoder = self.encoder
        centroids = {}
        rep = reduced_embedding
        rep_rep = encoder.encode(filtered_candidate_vocab) # do we need to DR to the same dimension as rep
        total = filtered_candidate_vocab

        globals()['frame'] = pd.DataFrame()
        globals()['frame']['text'] = text
        globals()['frame']['label'] = cluster_labels

        for m in list(set(cluster_labels)):
            index = frame[frame['label'] == m].index
            subset = rep[index]
            centroid = np.mean(rep[index], axis=0)
            centroids[m] = centroid

        centroid_keywords = {}
        for key in centroids.keys():
            centroid = centroids[key]
            similarity = cosine_similarity([centroid], rep_rep)
            centroid_keyword_index = similarity[0].argsort()[-3:][::-1]
            centroid_keywords[key] = [total[i] for i in centroid_keyword_index]

        return centroid_keywords

    def pipeline(self, dimension=5, clustering_method='hdbscan', min_cluster_size=2, n_clusters=5):
        embedding = self.encoding()
        reduced_embedding = self.DR(embedding, dimension=dimension)
        globals()['reduced_embedding'] = reduced_embedding
        if clustering_method == 'agglomerative':
            cluster_labels, _ = self.agglomerative_clustering(reduced_embedding, n_clusters=n_clusters)
        if clustering_method == 'hdbscan':
            cluster_labels, _ = self.clustering(reduced_embedding, min_cluster_size=min_cluster_size)

        candidate_vocab = self.build_candidates()
        centroid_keywords = self.centroid(reduced_embedding, candidate_vocab, cluster_labels)
        return cluster_labels, centroid_keywords


documents = []
with open('text.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(line.strip())  # Remove leading/trailing whitespace

documents = documents[:1000]
documents = [doc for doc in documents if doc is not None]
print("length of document", len(documents))

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
topic_modeling = TopicModeling(documents, model_name=MODEL_NAME, self_sim_threshold=0.5)
cluster_labels, centroid_keywords = topic_modeling.pipeline(clustering_method='hdbscan', min_cluster_size=2)
for cluster_id, keywords in centroid_keywords.items():
    print(f"Cluster {cluster_id}: {', '.join(keywords)}")
