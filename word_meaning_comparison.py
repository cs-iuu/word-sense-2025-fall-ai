
import numpy as np
import random
import re
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# Files (Make sure these files exist in the same folder)
MN_FILES = ['Bible_NT.mn.txt']
EN_FILES = ['Bible_NT.en-kjv.txt']
# Target Words to Analyze
TARGET_PAIRS = [
    ("бурхан", "god"),
    ("гэрэл", "light"),
    ("хүн", "man")
]
def load_corpus(file_list):
    """
    Reads files with automatic encoding detection (UTF-8 vs UTF-16).
    Fixes the '0 matches' bug caused by Windows file formats.
    """
    full_text = ""
    for filename in file_list:
        if not os.path.exists(filename):
            print(f" АЛДАА: Файл олдсонгүй -> {filename}")
            continue

        # Try 1: Read as UTF-8
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for "Null Bytes" (common sign of wrong encoding)
                if '\x00' in content: 
                    raise ValueError("Null bytes detected (likely UTF-16)")
                full_text += content + " "
                print(f" Файл: {filename} (Format: UTF-8)")
        except (UnicodeDecodeError, ValueError):
            # Try 2: Read as UTF-16 (Common on Windows)
            try:
                with open(filename, 'r', encoding='utf-16') as f:
                    full_text += f.read() + " "
                print(f" Файл: {filename} (Format: UTF-16)")
            except Exception as e:
                print(f" Файлыг уншиж чадсангүй: {filename}. Error: {e}")

    # Clean text: Lowercase and remove newlines
    cleaned_text = re.sub(r'[\r\n]+', ' ', full_text).lower()
    return cleaned_text

print("--- Data Loading ---")
MN_BIBLE_TEXT = load_corpus(MN_FILES)
EN_BIBLE_TEXT = load_corpus(EN_FILES)

# DEBUG: Verify we actually have text now
if len(MN_BIBLE_TEXT) > 100:
    print(f"\n[DEBUG] MN Text Sample: {MN_BIBLE_TEXT[:100]}...")
    print(f"[DEBUG] EN Text Sample: {EN_BIBLE_TEXT[:100]}...\n")


EMBEDDING_MODEL_NAME = "xlm-roberta-base"

try:
    print(f" Моделе ачаалж байна: {EMBEDDING_MODEL_NAME}...")
    EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMB_MODEL.to(DEVICE)
    print(f" XLM-R загвар амжилттай ачаалагдсан. (Төхөөрөмж: {DEVICE})")
except Exception as e:
    print(f"\n АЛДАА: Модель ачаалахад алдаа гарлаа: {e}")
    raise


def get_contextual_embedding(sentence, target_word):
    """Extracts embedding. Handles tokenization logic."""
    inputs = EMB_TOKENIZER(sentence, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        outputs = EMB_MODEL(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)

    ids = inputs['input_ids'].squeeze(0).tolist()
    word_tokens = EMB_TOKENIZER.convert_ids_to_tokens(ids)
    
    target_indices = []
    
    # Improved Token Matching
    for i, token in enumerate(word_tokens):
        if i == 0 or i == len(word_tokens) - 1: continue 
        
        # Clean token (remove ' ' and special chars)
        clean_token = token.replace(' ', '').lower()
        
        # Check if the token is part of the target word
        if clean_token in target_word or target_word in clean_token:
            target_indices.append(i)

    if not target_indices:
        return None

    # Get average of identified tokens
    start = target_indices[0]
    end = target_indices[-1]
    
    target_embs = last_hidden[start:end+1, :]
    return target_embs.mean(dim=0).cpu().numpy()

def collect_embeddings(corpus_text, target_word):
    """Scans corpus using relaxed matching."""
    if not corpus_text: return [], []
    
    # Split by sentence delimiters
    sentences = re.split(r'[.!?;\n]', corpus_text)
    
    contexts = []
    embeddings = []
    count = 0 
    MAX_SAMPLES = 150 
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 3: continue 
        if count >= MAX_SAMPLES: break
        
        # RELAXED CHECK: just 'in' string
        if target_word in sent:
            try:
                emb = get_contextual_embedding(sent, target_word)
                if emb is not None:
                    embeddings.append(emb)
                    contexts.append(sent)
                    count += 1
            except:
                continue
                
    return contexts, np.array(embeddings)

def cluster_senses(embeddings, k=2):
    if len(embeddings) < k: k = max(1, len(embeddings))
    if k == 0: return None, []
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
    return kmeans, kmeans.labels_

def compare_clusters(mo_centers, en_centers):
    return cosine_similarity(mo_centers, en_centers)

def print_cluster_samples(sentences, labels, cluster_id, n=3):
    indices = [i for i, label in enumerate(labels) if label == cluster_id]
    if not indices:
        print("    (No sentences)")
        return
    sample_indices = random.sample(indices, min(len(indices), n))
    for idx in sample_indices:
        clean_sent = sentences[idx].replace('\n', ' ')
        print(f"    - ...{clean_sent[:100]}...")


if __name__ == "__main__":
    if len(MN_BIBLE_TEXT) < 100 or len(EN_BIBLE_TEXT) < 100:
        print("\n Гүйцэтгэл цуцлагдлаа. Текст уншиж чадсангүй.")
    else:
        print("\n" + "="*60)
        print(" ЭХЛЭЛ: ХЭЛ ХООРОНДЫН УТГЫГ УЯЛДУУЛАХ (WSI)")
        print("="*60)

        # SETTINGS
        K_CLUSTERS = 3
        # Threshold for determining a "Match". 
        # If score > 0.99, we count it as a match. Otherwise, unmatch.
        # You can lower this to 0.985 if you want more matches.
        MATCH_THRESHOLD = 0.9900 

        for mo_word, en_word in tqdm(TARGET_PAIRS, desc="Processing Words"):
            print(f"\n\n{'='*40}")
            print(f"🔹 ШИНЖИЛЖ БУЙ ҮГС: '{mo_word}' (MN) <--> '{en_word}' (EN)")
            print(f"{'='*40}")
            
            # 1. Collect Embeddings
            mo_sents, mo_embs = collect_embeddings(MN_BIBLE_TEXT, mo_word)
            en_sents, en_embs = collect_embeddings(EN_BIBLE_TEXT, en_word)
            
            if len(mo_embs) < 5 or len(en_embs) < 5:
                print(f" Өгөгдөл дутуу (MN:{len(mo_embs)}, EN:{len(en_embs)}). Алгаслаа.")
                continue

            # 2. Clustering
            mo_kmeans, mo_labels = cluster_senses(mo_embs, k=K_CLUSTERS)
            en_kmeans, en_labels = cluster_senses(en_embs, k=K_CLUSTERS)

            # 3. Print Samples (Contexts)
            print(f"\n  --- 🇲🇳 MN Senses (Contexts) ---")
            for i in range(K_CLUSTERS):
                print(f"  [MN Cluster {i}]")
                print_cluster_samples(mo_sents, mo_labels, i, n=1)

            print(f"\n  --- 🇬🇧 EN Senses (Contexts) ---")
            for i in range(K_CLUSTERS):
                print(f"  [EN Cluster {i}]")
                print_cluster_samples(en_sents, en_labels, i, n=1)

            # 4. Compare & Count Matches
            print(f"\n  ---  SUMMARY REPORT (Threshold: {MATCH_THRESHOLD}) ---")
            # This line generates the 3x3 matrix of scores comparing every MN sense to every EN sense
            sim_matrix = compare_clusters(mo_kmeans.cluster_centers_, en_kmeans.cluster_centers_)
            
            matched_count = 0
            unmatched_count = 0

            print(f"  {'Status':<12} | {'MN Cluster':<10} | {'Best EN Match':<13} | {'Score'}")
            print("-" * 55)

            for mn_idx in range(K_CLUSTERS):
                best_en_idx = np.argmax(sim_matrix[mn_idx])
                score = sim_matrix[mn_idx][best_en_idx]
                
                # Check against Threshold
                if score >= MATCH_THRESHOLD:
                    status = " MATCHED"
                    matched_count += 1
                else:
                    status = " UNMATCHED"
                    unmatched_count += 1
                
                print(f"  {status:<12} | {mn_idx:<10} | {best_en_idx:<13} | {score:.4f}")

            # 5. Final Stats for this word
            print(f"\n   FINAL STATS for '{mo_word}':")
            print(f"     Total Senses Detected: {K_CLUSTERS}")
            print(f"     Successfully Matched:  {matched_count}")
            print(f"     Unmatched / Different: {unmatched_count}")

        print("\n Гүйцэтгэл амжилттай дууслаа.")
