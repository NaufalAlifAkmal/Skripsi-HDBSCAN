import os
import re
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse, unquote
from transformers import BertTokenizer, BertModel
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize
from decoder import parse_dec_file_to_dataframe
import io
import sys
# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Select GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model ONCE globally for efficiency
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL = BertModel.from_pretrained("bert-base-uncased").to(device)
MODEL.eval()

def extract_unique_urls(df):
    """
    Extract unique URLs from the dataframe and mask numeric values.

    Args:
        df (pd.DataFrame): Dataframe containing a 'url' column with log URLs.

    Returns: 
        list: Unique URLs with numeric sequences replaced by <NUM>.
    """
    result = df['url'].unique()
    result = [re.sub(r'\d+', '<NUM>', url) for url in result]
    return list(set(result))

def split_url_tokens(url):
    """
    Tokenize a URL by splitting its path and query string.

    Args:
        url (str): A URL string to tokenize.

    Returns:
        list: List of token strings extracted from the URL.
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)
    query = unquote(parsed.query)
    delimiters = r"[\/\?\&\=]"
    tokens = re.split(delimiters, path.strip("/")) + re.split(delimiters, query)
    return [tok for tok in tokens if tok]

def generate_url_embeddings(url_list, batch_size=16):
    """
    Generate BERT embeddings for a list of URLs.

    Args:
        url_list (list): List of preprocessed URL strings.
        batch_size (int, optional): Number of URLs to process in each batch. Defaults to 16.

    Returns:
        np.ndarray: Array of embedding vectors for each URL.
    """
    embeddings = []
    for i in range(0, len(url_list), batch_size):
        batch = url_list[i:i+batch_size]
        inputs = TOKENIZER(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = MODEL(**inputs)
        # Average token embeddings
        batch_emb = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_emb.cpu())
    return torch.cat(embeddings, dim=0).numpy()

def cluster_urls_from_log(df, out_path, min_cluster_size=5, min_samples=3):
    """
    Cluster URLs from log data using BERT embeddings and HDBSCAN.

    Args:
        df (pd.DataFrame): Dataframe containing URLs for clustering.
        out_path (str): Path for saving output CSV and TXT results.
        min_cluster_size (int): Minimum size of clusters for HDBSCAN. Defaults to 5.
        min_samples (int): Minimum samples in a neighborhood for HDBSCAN. Defaults to 3.

    Returns:
        None
    """
    # Step 1: Extract unique URLs
    unique_urls = extract_unique_urls(df)
    print(f"Output of: unique_urls")
    print(unique_urls)

    # Step 2: Tokenize each URL into text
    tokenized_urls = [" ".join(split_url_tokens(url)) for url in unique_urls]
    print(f"Output of: tokenized_urls")
    print(tokenized_urls)

    # Step 3: Convert URLs to embeddings
    embeddings = generate_url_embeddings(tokenized_urls)
    embeddings = normalize(embeddings)
    print(f"Output of: embeddings")
    print(embeddings)

    # Step 4: Cluster embeddings using HDBSCAN
    hdbscan = HDBSCAN(min_cluster_size=3, min_samples=1, metric='cosine')
    labels = hdbscan.fit_predict(embeddings)
    print(f"Output of: labels")
    print(labels)

    # Step 5: Group URLs by cluster (including noise points as cluster -1)
    unique_labels = set(labels)
    clustered_urls = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        clustered_urls[label].append(unique_urls[idx])

    # Step 6: Save clusters to a text file
    with open(f"{out_path}.txt", "w", encoding="utf-8") as f:
        for cluster, urls in sorted(clustered_urls.items()):
            if cluster == -1:
                f.write(f"\nNoise Points (Outliers):\n")
            else:
                f.write(f"\nCluster {cluster}:\n")
            for url in urls:
                f.write(f"  {url}\n")
        
        # Add clustering summary
        n_clusters = len([label for label in unique_labels if label != -1])
        n_noise = len(clustered_urls.get(-1, []))
        f.write(f"\n--- Clustering Summary ---\n")
        f.write(f"Total clusters found: {n_clusters}\n")
        f.write(f"Noise points (outliers): {n_noise}\n")
        f.write(f"Total URLs processed: {len(unique_urls)}\n")

    # Step 7: Save results to CSV
    df_label = pd.DataFrame({"masked": unique_urls, "cluster": labels}).sort_values(by='cluster')
    df_label.to_csv(out_path, index=False, encoding="utf-8")
    
    # Print clustering summary
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = len(clustered_urls.get(-1, []))
    print(f"HDBSCAN Clustering completed:")
    print(f"   Total clusters found: {n_clusters}")
    print(f"   Noise points (outliers): {n_noise}")
    print(f"   Total URLs processed: {len(unique_urls)}")
    print(f"   Results saved to: {out_path}")
    print(f"   Detailed clusters saved to: {out_path}.txt")

if __name__ == "__main__":
    """
    CLI entry point for NGINX log clustering tool using HDBSCAN.

    Example:
        python main.py inputs/sample.log outputs/clusters.csv --min-cluster-size 5 --min-samples 3
    """
    parser = argparse.ArgumentParser(description="Cluster NGINX log URLs using BERT embeddings and HDBSCAN.")
    parser.add_argument("in_file", help="NGINX log file")
    parser.add_argument("out_file", help="The labeled NGINX CSV file")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                       help="Minimum size of clusters for HDBSCAN (default: 5)")
    parser.add_argument("--min-samples", type=int, default=3,
                       help="Minimum samples in a neighborhood for HDBSCAN (default: 3)")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.in_file):
        print(f"❌ File not found: '{args.in_file}'")
        sys.exit(1)

    # Validate output directory
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        print(f"❌ Output directory not found: '{out_dir}'")
        sys.exit(1)
    
    # Load and process log file
    df = parse_dec_file_to_dataframe(args.in_file)
    print(f"Loaded {len(df)} rows from {args.in_file}")

    # Run clustering process
    cluster_urls_from_log(df, args.out_file, args.min_cluster_size, args.min_samples)