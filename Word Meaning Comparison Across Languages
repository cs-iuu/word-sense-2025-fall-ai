import numpy as np
import random
import json
import re
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
MN_FILES = ['Bible_NT.mn.txt']
EN_FILES = ['Bible_NT.en-kjv.txt']
def load_corpus(file_list):
    full_text = ""
    for filename in file_list:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                full_text += f.read() + " "
            print(f" Файл: {filename} амжилттай ачаалагдлаа.")
        except FileNotFoundError:
            print(f" АЛДАА: Файл: {filename} олдсонгүй. Файлын нэр эсвэл зам зөв эсэхийг шалгана уу.")
            continue
   
    cleaned_text = re.sub(r'[\r\n]+', ' ', full_text).lower()
    return cleaned_text

MN_BIBLE_TEXT = load_corpus(MN_FILES)
EN_BIBLE_TEXT = load_corpus(EN_FILES)
EMBEDDING_MODEL_NAME = "xlm-roberta-base"
try:
    EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    # GPU байгаа эсэхийг шалгана
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMB_MODEL.to(DEVICE)
    print(f"\n XLM-R загвар амжилттай ачаалагдсан. (Төхөөрөмж: {DEVICE})")
except Exception as e:
    print(f"\n АЛДАА: XLM-R загвар ачаалахад алдаа гарлаа: {e}")
    raise

# WSI хийх үгсийн хослол (Таны корпусын давтамж өндөр үгс)
TARGET_PAIRS = [
    ("бурхан", "god"),
    ("гэрэл", "light"),
    ("ус", "water"),
    ("газар", "earth"),
    ("хүн", "man")
]
def get_contextual_embedding(sentence, target_word):
    """Тухайн өгүүлбэр дэх сонгосон үгний контекстуал векторуудын дунджийг буцаана."""
    # Токенжуулалт
    inputs = EMB_TOKENIZER(sentence, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = EMB_MODEL(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)

    ids = inputs['input_ids'].squeeze(0).tolist()
    word_tokens = EMB_TOKENIZER.convert_ids_to_tokens(ids)
   
    target_start_index = -1
    target_end_index = -1
   
    current_word = ""
    for i, token in enumerate(word_tokens):
        if i == 0 or i == len(word_tokens) - 1: # [CLS] and [SEP]
            continue
       
        token_text = token[1:] if token.startswith(" ") else token
       
        if token.startswith(" "):
            if current_word == target_word:
                target_end_index = i - 1
                break
            current_word = token_text
        else:
            current_word += token_text
       
        if current_word == target_word and target_start_index == -1:
             target_start_index = i
             target_end_index = i

    if target_start_index != -1 and target_end_index == -1:
        target_end_index = target_start_index

    if target_start_index != -1:
        target_embs = last_hidden[target_start_index:target_end_index+1, :]
        return target_embs.mean(dim=0).cpu().numpy()
       
    return None

def collect_embeddings(corpus_text, target_word):
    """Корпусаас өгүүлбэрүүдийг цуглуулж, контекстуал векторуудыг үүсгэнэ."""
    sentences = [s.strip() for s in corpus_text.split('.') if s.strip()]
   
    contexts = []
    embeddings = []
    for sent in sentences:
        if target_word in sent.lower():
            emb = get_contextual_embedding(sent, target_word)
            if emb is not None:
                embeddings.append(emb)
                contexts.append(sent)
               
    return contexts, np.array(embeddings)

def cluster_senses(embeddings, k=2):
    """K-Means ашиглан утгын багцлалт хийж, төв цэгүүд болон шошгуудыг буцаана."""
    if len(embeddings) < k:
        k = max(1, len(embeddings))
       
    if k == 0:
        return None, []
       
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
    return kmeans, kmeans.labels_

def compare_clusters(mo_centers, en_centers):
    """Монгол ба Англи утгын төв цэгүүдийн хоорондын косинус төстэй байдлыг тооцоолно. """
    sim = cosine_similarity(mo_centers, en_centers)
    return sim
if not MN_BIBLE_TEXT or not EN_BIBLE_TEXT:
    print("\n Гүйцэтгэл цуцлагдлаа. Корпус ачааллахад алдаа гарсан.")
else:
    FINAL_RESULTS = {}

    print("\n--- Эхлэл: Хэл Хоорондын Утгыг Уялдуулах ---")

    for mo_word, en_word in tqdm(TARGET_PAIRS, desc="Үгсийн хослолоор гүйцэтгэж байна"):
       
        # 1. Embeddings цуглуулах
        mo_sents, mo_embs = collect_embeddings(MN_BIBLE_TEXT, mo_word)
        en_sents, en_embs = collect_embeddings(EN_BIBLE_TEXT, en_word)
