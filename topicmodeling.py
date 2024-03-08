from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
import hdbscan
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TopicModeling:
    def __init__(self, text, model='all-mpnet-base-v2', device='cuda', self_sim_threshold=0.5):
        self.corpus = text
        self.encoder = SentenceTransformer(model, device=device)
        self.tokenizer = self.encoder.tokenizer
        self.self_sim_threshold = self_sim_threshold

    def encoding(self):
        embedding = self.encoder.encode(self.corpus)
        globals()['embedding'] = embedding
        return embedding

    def DR(self, embedding, dimension=5):
        reducer = umap.UMAP(random_state=42, n_components=dimension)
        embedding = reducer.fit_transform(embedding)
        return embedding

    def clustering(self, embedding, min_cluster_size=2):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embedding)
        outlier_scores = clusterer.outlier_scores_
        return cluster_labels, outlier_scores

    def agglomerative_clustering(self, embedding, n_clusters=5):
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embedding)
        cluster_labels = clustering.labels_
        return cluster_labels, _

    def self_similarity(self, all_hidden_states, token_list, inference_list):
        ss_score = {}

        temp = all_hidden_states[-1]

        for token in tqdm(token_list, desc='Token Progress'):
            token_embeddings = []

            for (sentence_index, token_index) in inference_list[token]:
                token_embeddings.append(np.array(temp[sentence_index][token_index]))

            token_embeddings = np.array(token_embeddings)
            sim_matrix = cosine_similarity(token_embeddings, token_embeddings)

            if len(sim_matrix) != 1:
                self_similarity = round((np.sum(sim_matrix) - len(sim_matrix)) / (len(sim_matrix) * (len(sim_matrix) - 1)), 3)
                ss_score[token] = self_similarity

        return ss_score

    def build_candidates(self, corpus, encoder):
        print("Tokenizing text with model:", encoder.model)
        encoded_input = self.tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')
        print("Finished Tokenizing:", encoder.model)
        tokenized = encoded_input['input_ids'].tolist()
        token_list = list(itertools.chain.from_iterable(tokenized))
        counter = Counter(token_list)

        index_count = [(index, count) for (index, count) in counter.most_common() if count >= 5]
        candidate_vocab = [index for (index, count) in counter.most_common() if count >= 5]

        all_hidden_states = encoder.model.get_all_hidden_states(encoder.encode(corpus))

        inference_list = {}
        for n in set(candidate_vocab):
            position_list = []
            for sen_index, sen in enumerate(tokenized):
                if n in sen:
                    token_index = sen.index(n)
                    position_list.append((sen_index, token_index))
            inference_list[n] = position_list

        ss_score = self.self_similarity(all_hidden_states, candidate_vocab, inference_list)
        filtered_candidate_vocab = [token for token, self_sim in ss_score.items() if self_sim >= self.self_sim_threshold]

        return filtered_candidate_vocab
        
    def centroid(self, embedding, candidate_vocab, cluster_labels):
        text = self.corpus
        encoder = self.encoder
        centroids = {}
        rep = embedding
        rep_rep = encoder.encode(candidate_vocab)
        total = candidate_vocab

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

        candidate_vocab = self.build_candidates(self.corpus, self.encoder)
        centroid_keywords = self.centroid(embedding, candidate_vocab, cluster_labels)
        return cluster_labels, centroid_keywords
    
documents = []
with open('text.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(line.strip())  # Remove leading/trailing whitespace


topic_modeling = TopicModeling(documents, self_sim_threshold=0.5)
cluster_labels, centroid_keywords = topic_modeling.pipeline(clustering_method='hdbscan', min_cluster_size=2)

for cluster_id, keywords in centroid_keywords.items():
    print(f"Cluster {cluster_id}: {', '.join(keywords)}")