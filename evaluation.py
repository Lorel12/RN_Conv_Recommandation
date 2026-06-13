import numpy as np
import torch


def evaluate_model(model, norm_adj, train_df, test_df, k: int = 20):
    """
    Phase 5 : Mesure de la qualité des recommandations (Recall@K et NDCG@K).

    model : instance de LightGCN (en mode eval).
    norm_adj : matrice d'adjacence normalisée (tenseur COO creux).
    train_df : DataFrame des interactions d'entraînement.
    test_df : DataFrame des interactions de test.
    k : nombre de recommandations Top-K.

    """
    model.eval()

    with torch.no_grad():
        users_final, items_final = model(norm_adj) # (U, d), (I, d)

    # Transfert vers CPU pour le traitement numpy
    users_emb = users_final.cpu().numpy()  # (U, d)
    items_emb = items_final.cpu().numpy() # (I, d)

    # dictio d'interactions 
    train_dict = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    test_dict  = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    recalls, ndcgs = [], []

    # Precalcul du vecteur IDCG pour chaque taille possible
    idcg_cache = np.array([
        sum(1.0 / np.log2(r + 2) for r in range(min(n, k)))
        for n in range(0, max(len(v) for v in test_dict.values()) + 2)
    ])

    for u, actual_positives in test_dict.items():
        if len(actual_positives) == 0:
            continue

        # Score, produit scalaire de l'user avec tous les articles
        scores = users_emb[u] @ items_emb.T            # (I,)

        # Masquage des articles déjà vus en entraînement
        if u in train_dict:
            scores[list(train_dict[u])] = -np.inf

        # Top-K recommandations
        top_k = np.argpartition(scores, -k)[-k:] # non triee
        top_k = top_k[np.argsort(scores[top_k])[::-1]] # triee par score desc

        # ─ Recall@K 
        hits = len(set(top_k) & actual_positives)
        recalls.append(hits / len(actual_positives))

        # ─ NDCG@K 
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, item in enumerate(top_k)
            if item in actual_positives
        )
        idcg = idcg_cache[len(actual_positives)]
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))