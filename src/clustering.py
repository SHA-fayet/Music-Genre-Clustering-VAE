import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def perform_clustering(latents, n_clusters=2):
    """
    Applies multiple clustering algorithms on the latent representation.
    
    Args:
        latents (np.array): The latent vectors (N, latent_dim).
        n_clusters (int): Number of target clusters (usually 2 for Western vs Latin).
        
    Returns:
        dict: A dictionary containing labels for 'kmeans', 'agg', and 'dbscan'.
    """
    results = {}
    
    # 1. K-Means (Baseline)
    # We use n_init=10 to ensure stable initialization
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results["kmeans"] = kmeans.fit_predict(latents)
    
    # 2. Agglomerative Clustering (Hierarchical)
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    results["agg"] = agg.fit_predict(latents)
    
    # 3. DBSCAN (Density-Based)
    # Heuristic: We estimate epsilon based on the standard deviation of the latent space.
    # If the latent space is normalized (standard normal), std is close to 1.
    eps_val = np.std(latents) * 0.5
    if eps_val == 0: 
        eps_val = 0.5
    
    dbscan = DBSCAN(eps=eps_val, min_samples=5)
    results["dbscan"] = dbscan.fit_predict(latents)
    
    return results