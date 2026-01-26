# Mongolian Dictionary
This is the top level repository for 2025 Fall AI Basics team project to build a Mongolian dictionary automatically.

**Goals:**
1. To create a partial, prototype Mongolian Dictionary containing a set of headwords with automatically generated word senses (clusters of usage) and sample context sentences extracted from a Mongolian text corpus (e.g., the Mongolian Bible or another large, available text).
2. To compare word meanings, usage patterns, and semantic relations with other languages (e.g., English) to reveal structural and linguistic insights.

# Bullet points
1. Imports

Loads tools for text processing, embeddings (XLM-RoBERTa), clustering (KMeans), and similarity (cosine).

2. Files & Target Words

Defines MN/EN Bible texts and word pairs to compare meanings across languages.

3. load_corpus

Reads text files (UTF-8 / UTF-16), fixes Windows encoding issues, cleans and lowercases text.

4. Model Loading

Loads XLM-RoBERTa and moves it to GPU if available to generate contextual embeddings.

**function** get_contextual_embedding

Extracts the average embedding of the target word from a sentence using token matching.

**function** collect_embeddings

Finds sentences containing the target word and collects their embeddings (max 150).


**function** cluster_senses

Groups embeddings into k sense clusters using K-Means.

**function** compare_clusters

Computes cosine similarity between MN and EN sense cluster centers.

**function** print_cluster_samples

Prints example sentences for each sense cluster.
