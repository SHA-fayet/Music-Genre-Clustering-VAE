from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
import numpy as np

def calculate_metrics(data, preds, true_labels=None):
    """
    Computes clustering metrics including Silhouette, NMI, ARI, and Purity.
    
    Args:
        data (np.array): Latent vectors (for Silhouette).
        preds (np.array): Predicted cluster labels.
        true_labels (np.array, optional): Ground truth labels (Western vs Latin).
        
    Returns:
        dict: Dictionary of calculated metric scores.
    """
    metrics = {}
    
    # 1. Silhouette Score (Internal Metric)
    # Checks how similar an object is to its own cluster compared to other clusters.
    # Only valid if we have > 1 cluster and < N clusters
    unique_labels = np.unique(preds)
    if len(unique_labels) > 1 and len(unique_labels) < len(data):
        metrics["Silhouette"] = silhouette_score(data, preds)
    else:
        metrics["Silhouette"] = 0.0
        
    # 2. Ground Truth Metrics (if labels available)
    if true_labels is not None:
        # NMI (Normalized Mutual Information)
        metrics["NMI"] = normalized_mutual_info_score(true_labels, preds)
        
        # ARI (Adjusted Rand Index)
        metrics["ARI"] = adjusted_rand_score(true_labels, preds)
        
        # Purity Calculation
        # Purity is the percent of the total number of data points that were classified correctly.
        cm = confusion_matrix(true_labels, preds)
        # Sum of the max value in each column / total samples
        purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        metrics["Purity"] = purity
        
    return metrics