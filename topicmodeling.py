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

class candidate():

    def __init__(self, text,
                 mode: str = 'n_gram',
                 ngram_range = (2,4),
                 feature_1gram: int = 2000,
                 feature_mgram: int = 10000,
                 punctuation: str = '[?!,.]'
                 ):
        self.corpus = np.array(text)
        self.mode = mode

        self.ngram_range = ngram_range
        self.feature_1gram = feature_1gram
        self.feature_mgram = feature_mgram
        self.punctuation = punctuation

    def lower(self):
        text = self.corpus
        text = text.astype('U')
        text = np.char.lower(text)

        return text

    def n_gram(self):
        ngram_range = self.ngram_range
        feature_1gram = self.feature_1gram
        feature_mgram = self.feature_mgram

        text = self.lower()
        vectorizer = CountVectorizer(max_features = feature_1gram)
        vectorizer2 = CountVectorizer(analyzer='word', ngram_range=ngram_range, max_features = feature_mgram)
        vectorizer.fit(text)
        vectorizer2.fit(text)
        vocab = vectorizer.get_feature_names_out()
        phrase = vectorizer2.get_feature_names_out()
        total = list(np.concatenate([vocab,phrase]))

        return total

    def sentence_level(self):

        text = self.corpus
        punctuation = self.punctuation

        sentence_list = []
        for index, doc in enumerate(text):
          doc = doc.strip()
          doc = re.split(punctuation,doc)
          doc = [sent.strip() for sent in doc]
          doc = list(filter(None, doc))
          sentence_list.extend(doc)
        return list(set(sentence_list))

    def document_level(self):

        text = list(self.corpus)
        return list(set(text))

    def build_vocab(self):
        if self.mode == 'n_gram':
             total = self.n_gram()
        elif self.mode == 'sentence_level':
             total = self.sentence_level()
        elif self.mode == 'document_level':
             total = self.document_level()

        return total


class TopicModeling:
    def __init__(self, text, model_name, dimension=5, device='cuda', self_sim_threshold=0.5, batch_size=32):
        self.corpus = text
        self.self_sim_threshold = self_sim_threshold
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dimension = dimension

    def encoding(self):
        print("Encoding text:")
        embedding = self.encoder.encode(self.corpus, show_progress_bar=True, batch_size=self.batch_size)
        print("Finished encoding")

        globals()['embedding'] = np.array(embedding)
        return embedding

    def self_similarity(self, last_hidden_state, token_list, inference_list):
        ss_score = {}

        for token in tqdm(token_list, desc='Token Progress'):
            token_embeddings = []
            for token_index in inference_list[token]:
                token_embedding = np.array(last_hidden_state[token_index])
                token_embeddings.append(token_embedding)
            token_embeddings = np.array(token_embeddings)
            if len(token_embeddings) > 0:
                sim_matrix = cosine_similarity([token_embeddings], [token_embeddings])
                self_similarity = round(
                    (np.sum(sim_matrix) - len(sim_matrix)) / (len(sim_matrix) * (len(sim_matrix) - 1)), 3)
                ss_score[token] = self_similarity

        return ss_score


    def build_candidates(self): # return candidate vocabulary and embeddings
        candidate_vocab = candidate(self.corpus, mode="n_gram", ngram_range=(1, 1)).build_vocab()
        print(f"Candidate vocabulary length: {len(candidate_vocab)}")
    
        if len(candidate_vocab) == 0:
            print("No candidate vocabulary generated. Check the input corpus or candidate class parameters.")
            return {}

        print("Encoding candidate vocabulary with model:")
        rep_rep = self.encoder.encode(candidate_vocab, show_progress_bar=True, batch_size=self.batch_size)

        inference_list = {}
        for i, token in enumerate(candidate_vocab):
            inference_list[token] = [(i, 0)]  # Each token corresponds to a single position in the encoded representation

        ss_score = self.self_similarity(rep_rep, list(inference_list.keys()), inference_list)
        filtered_candidate_vocab = [token for token, self_sim in ss_score.items() if self_sim >= self.self_sim_threshold]
        print(f"filtered_candidate_vocab: {filtered_candidate_vocab}")
        
        rep_candidate = {token: rep_rep[i] for token, i in inference_list.items() if token in filtered_candidate_vocab}
        print(f"rep_candidate: {rep_candidate}")

        return rep_candidate # dic store filtered_candidates as keys and corresponding embeddings as values

    def DR(self, embedding):
        dimension = self.dimension
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

    def centroid(self, reduced_embedding, filtered_candidate_vocab, reduced_rep_candidate, cluster_labels):
        text = self.corpus
        centroids = {}
        rep = reduced_embedding
        total = filtered_candidate_vocab

        globals()['frame'] = pd.DataFrame()
        globals()['frame']['text'] = text
        globals()['frame']['label'] = cluster_labels

        for m in list(set(cluster_labels)):
            index = frame[frame['label'] == m].index
            centroid = np.mean(rep[index], axis=0)
            centroids[m] = centroid

        centroid_keywords = {}
        for key in centroids.keys():
            centroid = centroids[key]
            similarity = cosine_similarity([centroid], reduced_rep_candidate)
            centroid_keyword_index = similarity[0].argsort()[-10:][::-1]
            #valid_indices = [i for i in centroid_keyword_index if i < len(total)]
            #centroid_keywords[key] = [total[i] for i in valid_indices]
            centroid_keywords[key] = [total[i] for i in centroid_keyword_index]

        return centroid_keywords

    def pipeline(self, clustering_method='hdbscan', min_cluster_size=2, n_clusters=5):
        embedding = self.encoding()
        reduced_embedding = self.DR(embedding)
        globals()['reduced_embedding'] = reduced_embedding
        if clustering_method == 'agglomerative':
            cluster_labels, _ = self.agglomerative_clustering(reduced_embedding, n_clusters=n_clusters)
        if clustering_method == 'hdbscan':
            cluster_labels, _ = self.clustering(reduced_embedding, min_cluster_size=min_cluster_size)

       
        filtered_candidate_vocab, rep_candidate = self.build_candidates().items()

        reduced_rep_candidate = self.DR(rep_candidate)
        centroid_keywords = self.centroid(reduced_embedding, filtered_candidate_vocab, reduced_rep_candidate, cluster_labels)
        return cluster_labels, centroid_keywords


documents = []
with open('text.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(line.strip())  # Remove leading/trailing whitespace

documents = documents[:1000]
documents = [doc for doc in documents if doc is not None]
print("length of document", len(documents))

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
topic_modeling = TopicModeling(documents, model_name=MODEL_NAME, self_sim_threshold=0)
cluster_labels, centroid_keywords = topic_modeling.pipeline(clustering_method='hdbscan', min_cluster_size=2)
for cluster_id, keywords in centroid_keywords.items():
    print(f"Cluster {cluster_id}: {', '.join(keywords)}")
