import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
import h5py
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Downloads ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Configuration & Setup ---
# YOUR FILE PATH (Keep this exactly as you had it)
file_path = r"C:\Users\Hitech\Desktop\VS\vs random\html word game\team\les_miserables.txt"

HDF5_EMBEDDINGS_FILE = 'corpus_embeddings.h5'
INDEX_FILE = 'corpus_index.pkl'
BATCH_SIZE = 16
MODEL_NAME = 'distilbert-base-uncased'

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU. Processing will be slower.")

# --- Load Model ---
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# --- Load & Clean Corpus ---
clean_corpus = []

# *** FIX IS HERE: Added encoding='utf-8' ***
try:
    with open(file_path, 'r', encoding='utf-8') as f: 
        for line in f:
            if line.strip(): # Skip empty lines
                clean_corpus.append(line.strip())
except UnicodeDecodeError:
    # Fallback if utf-8 fails (sometimes files are latin-1)
    print("⚠️ UTF-8 failed, trying Latin-1...")
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.strip():
                clean_corpus.append(line.strip())
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found. Please check the path.")
    clean_corpus = ["I went to the bank.", "The river bank was muddy."] 

N_SENTENCES = len(clean_corpus)
print(f"Loaded {N_SENTENCES} sentences.")

CONTENT_TAGS_PREFIX = ('NN', 'VB', 'JJ', 'RB')
STOP_WORDS = set(stopwords.words('english'))

# Global list to collect index data
index_data = []

# --- Function Definitions ---

def process_corpus_general_batched(corpus, model, tokenizer, batch_size, device, h5f):
    """
    Processes the corpus in batches for fast BERT inference and indexes
    all content words, writing embeddings directly to the h5f disk file.
    """
    global index_data

    # 1. Chunk the entire corpus into batches
    batched_corpus = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
    global_sent_idx = 0

    print(f"Processing {len(corpus)} sentences in {len(batched_corpus)} batches of size {batch_size}...")

    # 2. Process each batch
    for batch_id, batch_texts in enumerate(batched_corpus):
        if batch_id % 10 == 0: # Print every 10th batch to reduce clutter
            print(f"Processing batch {batch_id + 1}/{len(batched_corpus)}...")

        # A. Get BERT Hidden States for the entire batch
        encoded_input = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded_input)
            full_hidden_states = outputs[0].cpu().numpy()

        # 3. Process each sentence's result from the batch for indexing
        for sent_in_batch, text in enumerate(batch_texts):
            input_ids_tensor = encoded_input['input_ids'][sent_in_batch].cpu()

            # --- DISK WRITE ---
            embedding_array = full_hidden_states[sent_in_batch]
            h5f.create_dataset(f'sent_{global_sent_idx}', data=embedding_array, compression="gzip")

            # 4. Identify Content Words using NLTK
            nltk_tokens = nltk.word_tokenize(text)
            tagged_tokens = nltk.pos_tag(nltk_tokens)

            content_words = [(word.lower(), tag) for word, tag in tagged_tokens
                             if word.isalpha() and word.lower() not in STOP_WORDS and tag.startswith(CONTENT_TAGS_PREFIX)]

            bert_tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist())

            # 5. Build the Index
            for word, _ in content_words:
                target_indices = [i for i, token in enumerate(bert_tokens)
                                  if word in token or word.capitalize() in token]

                if target_indices:
                    index_data.append({
                        'target_word': word,
                        'sentence_id': global_sent_idx,
                        'token_indices': target_indices,
                        'sentence': text
                    })

            global_sent_idx += 1

def get_target_vectors_from_store(target_word, index_df):
    """
    Retrieves the contextualized BERT vectors for all occurrences of a target word,
    reading the embedding data directly from the HDF5 file.
    """
    matches = index_df[index_df['target_word'] == target_word.lower()]

    if matches.empty:
        return []

    target_vectors = []

    with h5py.File(HDF5_EMBEDDINGS_FILE, 'r') as hf:
        for _, row in matches.iterrows():
            sent_id = row['sentence_id']
            token_indices = row['token_indices']

            try:
                full_sent_embedding = hf[f'sent_{sent_id}'][()]
            except KeyError:
                print(f"Warning: Dataset 'sent_{sent_id}' not found in HDF5 file.")
                continue

            word_vector = np.mean(full_sent_embedding[token_indices], axis=0)
            target_vectors.append(word_vector)

    return target_vectors

def find_optimal_k_and_cluster(X, max_k=5):
    """
    Finds the optimal K using Silhouette Score and performs K-means.
    """
    n_instances = X.shape[0]

    if n_instances < 2:
        print(f"   --> Warning: Only {n_instances} instance(s) found. Cannot cluster.")
        return 1, np.zeros(n_instances, dtype=int)

    k_range = range(2, min(max_k, n_instances - 1) + 1)

    if len(k_range) == 0:
        print(f"   --> Warning: Only {n_instances} instances. Defaulting to K=1.")
        return 1, np.zeros(n_instances, dtype=int)

    best_k = k_range[0]
    best_score = -1.0

    print(f"   --> Testing K in range {list(k_range)}...")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    final_labels = kmeans.fit_predict(X)

    print(f"   --> Optimal K determined: {best_k} (Silhouette: {best_score:.4f})")
    return best_k, final_labels


# --- EXECUTION STAGE 1: PROCESSING & INDEXING ---
print(f"--- Starting Stage 1: Indexing All Content Words for {N_SENTENCES} Sentences ---")
start_time = time.time()

# Cleanup old files
try:
    os.remove(HDF5_EMBEDDINGS_FILE)
    os.remove(INDEX_FILE)
except OSError:
    pass

# Open HDF5 and run processing
h5f = h5py.File(HDF5_EMBEDDINGS_FILE, 'w')
process_corpus_general_batched(clean_corpus, model, tokenizer, BATCH_SIZE, device, h5f)
h5f.close()

# Save Index
index_df = pd.DataFrame(index_data)
index_df.to_pickle(INDEX_FILE)

print(f"Processing complete. Index saved to {INDEX_FILE}")
print(f"Time taken for Stage 1 (BERT Inference): {time.time() - start_time:.2f} seconds.")
print(f"Index created for {len(index_df)} instances of ALL content words.")


# --- EXECUTION STAGE 2: CLUSTERING ---
print("\n--- Starting Stage 2: Efficient Sense Induction from Disk ---")

# Load Index
try:
    index_df = pd.read_pickle(INDEX_FILE)
    print(f"✅ Loaded index with {len(index_df)} word occurrences.")
except FileNotFoundError:
    print(f"🛑 Error: Index file '{INDEX_FILE}' not found. Did Stage 1 complete successfully?")
    exit()

# Filter for words with enough data
word_counts = index_df.groupby('target_word').size()
plausible_words = word_counts[word_counts >= 2].index.tolist()
print(f"Found {len(plausible_words)} words with 2 or more instances for clustering.")

# Words to Analyze
WORDS_TO_ANALYZE = ["man", "life", "light", "spirit", "son", "body", "bank"]

for word in WORDS_TO_ANALYZE:
    print(f"\nProcessing word: {word}")
    run_start = time.time()

    # A. Retrieve vectors
    X_list = get_target_vectors_from_store(word, index_df)

    if not X_list:
        print(f"  Skipping '{word}': No instances found or retrieval failed.")
        continue

    X = np.array(X_list)
    sentences = index_df[index_df['target_word'] == word.lower()]['sentence'].tolist()

    # B. Cluster
    optimal_k, labels = find_optimal_k_and_cluster(X, max_k=5)

    # C. Display
    sense_clusters = {i: [] for i in range(optimal_k)}
    for sentence, label in zip(sentences, labels):
        sense_clusters[label].append(sentence)

    run_end = time.time()
    print(f"## 🎯 Induced Senses for '{word}' (Run Time: {run_end - run_start:.4f}s) ##")

    for i, sentences_in_sense in sense_clusters.items():
        print(f"--- Sense Cluster {i+1} ({len(sentences_in_sense)} instances) ---")
        for j, sentence in enumerate(sentences_in_sense[:3]):
            print(f"  - {sentence}")
        if len(sentences_in_sense) > 3:
             print("  - ... (more instances)")

    print("-" * 20)


