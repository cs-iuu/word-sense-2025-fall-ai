import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import sys

# --- 1. CONFIGURATION AND DATA LOADING (FINAL REVISION) ---
# IMPORTANT: Check the path. If data.csv is in the same folder as this script, "data.csv" is correct.
CSV_PATH = "data.csv" 
HEADWORD = 'газар' # Example headword for analysis. Change this as needed.
N_CLUSTERS = 5     # Number of word senses (clusters) to automatically discover.

try:
    print(f"Loading data from: {CSV_PATH}")
    # Use the same encoding as your original local file
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
except FileNotFoundError:
    print(f"ERROR: File not found at {os.path.abspath(CSV_PATH)}")
    print("Please check the CSV_PATH variable in the script.")
    sys.exit(1)

# *** FIX: Adapt to the single-column data structure found in the last error ***
original_cols = df.columns.tolist()
print(f"Original columns: {original_cols}")

if not original_cols:
    print("FATAL ERROR: CSV file is empty or incorrectly formatted (no headers found).")
    sys.exit(1)

# Use the first (and likely only) column header
single_col_name = original_cols[0].strip()
df.columns = [single_col_name]

print(f"Using single data column (containing all entries): '{single_col_name}'")

# All the text data (Entries and Contents) is assumed to be in this single column.
raw_entries = df[single_col_name].fillna("").astype(str).tolist()

print(f"Successfully loaded {len(raw_entries)} raw entries from the single column.")


# --- 2. CLEANING AND TOKENIZATION ---
def clean(text):
    """Cleans text, retaining Mongolian and English characters."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^A-Za-zА-Яа-яӨҮөүЁёЇїЙйһ ]', ' ', text)
    return text.lower()

w2v_sentences = [] # Tokenized sentences (list of lists of tokens)
bert_sentences = [] # Raw cleaned sentences (list of strings)

for combined_text in raw_entries:
    cleaned = clean(combined_text).strip()
    toks = cleaned.split()
    
    if len(toks) >= 2:
        w2v_sentences.append(toks)
        bert_sentences.append(" ".join(toks))

# Ensure enough samples for Word2Vec training
if not w2v_sentences:
     print("ERROR: No valid entries found after cleaning. Exiting.")
     sys.exit(1)

while len(w2v_sentences) < 5:
    w2v_sentences += w2v_sentences
    bert_sentences += bert_sentences

print(f"✔ Final entries for processing: {len(bert_sentences)}")


# --- 3. FEATURE ENGINEERING: WORD2VEC (Context-Independent) ---
print("\n--- 3. Training Word2Vec Model (Context-Independent Features) ---")
w2v_model = Word2Vec(
    sentences=w2v_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=20
)

def get_w2v_sentence_vector(tokens, model):
    """Averages the vectors of all words in an entry."""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# Generate vectors for the entries used for BERT
w2v_vectors = np.array([
    get_w2v_sentence_vector(s.split(), w2v_model) for s in bert_sentences
])
print(f"✔ Word2Vec entry vectors shape: {w2v_vectors.shape}")


# --- 4. FEATURE ENGINEERING: BERT (Contextual Feature) ---
BERT_MODEL = "paraphrase-MiniLM-L6-v2"
print(f"\n--- 4. Encoding Entries with BERT Model: {BERT_MODEL} (Contextual Features) ---")

bert_model = SentenceTransformer(BERT_MODEL)
bert_vectors = bert_model.encode(bert_sentences, show_progress_bar=True)
bert_vectors = np.array(bert_vectors)

print(f"✔ BERT entry vectors shape: {bert_vectors.shape}")


# --- 5. SEMANTIC ANALYSIS AND COMPARISON ---
n_samples = w2v_vectors.shape[0]
n_comp = min(w2v_vectors.shape[1], bert_vectors.shape[1]) 

w2v_final = w2v_vectors[:n_samples]
bert_final = bert_vectors[:n_samples]

print(f"\n--- 5. Comparison: Reducing vectors to {n_comp} dimensions for Cosine Similarity ---")

# 5a. Reduce dimensions for comparison
pca_w2v = PCA(n_components=n_comp)
w2v_reduced = pca_w2v.fit_transform(w2v_final)

pca_bert = PCA(n_components=n_comp)
bert_reduced = pca_bert.fit_transform(bert_final)

# 5b. Cosine Similarity (Goal 2: Comparison of usage patterns)
sim_scores_diagonal = np.diag(cosine_similarity(w2v_reduced, bert_reduced)) 
print(f"✔ Average Semantic Feature Similarity (W2V vs BERT): {np.mean(sim_scores_diagonal):.4f}")


# --- 6. WORD SENSE CLUSTERING (Goal 1: Automatic Senses) ---
print(f"\n--- 6. Clustering BERT Embeddings into {N_CLUSTERS} Senses (K-Means) ---")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(bert_final)
print("✔ Clustering complete.")


# --- 7. VISUALIZATION ---
# Reduce BERT embeddings to 2D for plotting
pca_plot = PCA(n_components=2)
bert_2d = pca_plot.fit_transform(bert_final)

plt.figure(figsize=(15, 7))

# Subplot 1: BERT vectors colored by the discovered sense cluster
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(bert_2d[:, 0], bert_2d[:, 1], 
                      c=clusters, cmap='Spectral', s=50, alpha=0.7)
plt.colorbar(scatter1, label="Word Sense Cluster")
plt.title(f"BERT Embeddings Clustered into {N_CLUSTERS} Senses", fontsize=10)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: BERT vectors colored by the W2V/BERT similarity score
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(bert_2d[:, 0], bert_2d[:, 1], 
                      c=sim_scores_diagonal, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter2, label="W2V vs BERT Cosine Similarity")
plt.title("BERT Embeddings Colored by Feature Agreement Score (Goal 2 Insight)", fontsize=10)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle("Feature Engineering Comparison and Dictionary Sense Discovery", fontsize=14)
plt.show()


# --- 8. PROTOTYPE DICTIONARY SENSE EXTRACTION ---
print(f"\n--- 8. PROTOTYPE DICTIONARY SENSES (HEADWORD: '{HEADWORD}') ---")
headword_senses = []

# Collect entries and their cluster ID where the headword appears
for i, entry_text in enumerate(bert_sentences):
    # Check if the HEADWORD is present in the cleaned, tokenized entry text
    if HEADWORD in entry_text.split():
        original_entry = raw_entries[i].strip() 
        headword_senses.append({
            'entry': original_entry,
            'cluster_id': clusters[i],
            'similarity_score': sim_scores_diagonal[i]
        })

if headword_senses:
    sense_df = pd.DataFrame(headword_senses)
    
    # Analyze the sense groups
    for sense_id in sorted(sense_df['cluster_id'].unique()):
        sense_group = sense_df[sense_df['cluster_id'] == sense_id]
        print(f"\nSENSE {sense_id} (N={len(sense_group)} entries) - Avg Sim: {sense_group['similarity_score'].mean():.3f}")
        for i, row in enumerate(sense_group.head(2).itertuples()):
            print(f"  Sample {i+1} (Sim: {row.similarity_score:.2f}): {row.entry[:100]}...")
else:
    print(f"Headword '{HEADWORD}' not found in the processed entries.")