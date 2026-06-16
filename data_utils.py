import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def load_movielens_data(data_dir='ml-latest-small', threshold=4.0):
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"Le fichier {ratings_path} est introuvable. "
        )

    df = pd.read_csv(ratings_path)

    # Conservation des interactions significatives
    df = df[df['rating'] >= threshold].copy()

    # Encodage des identifiants en index continus
    user_mapping = {uid: i for i, uid in enumerate(df['userId'].unique())}
    item_mapping = {iid: i for i, iid in enumerate(df['movieId'].unique())}

    df['user_idx'] = df['userId'].map(user_mapping)
    df['item_idx'] = df['movieId'].map(item_mapping)

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    np.random.seed(42)
    train_rows, test_rows = [], []

    for _, group in df.groupby('user_idx'):
        group = group.sample(frac=1, random_state=42)   
        n = len(group)
        if n < 2:
            # Pas assez d'interactions, tout en entraenement
            train_rows.append(group)
            continue
        split = max(1, int(0.2 * n))                    
        test_rows.append(group.iloc[:split])
        train_rows.append(group.iloc[split:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    print(f"  Interactions totales : {len(df):,}  →  "
          f"Train : {len(train_df):,}  |  Test : {len(test_df):,}")

    return train_df, test_df, num_users, num_items, user_mapping, item_mapping


def create_adjusted_adjacency(train_df, num_users, num_items):
    """
    Phase 2: Construction du graphe biparti et génération de la matrice
    d'adjacence normalisée D^{-1/2} A D^{-1/2} sous forme de tenseur
    COO creux PyTorch.
    """
    user_np = train_df['user_idx'].values
    item_np = train_df['item_idx'].values
    ones    = np.ones(len(train_df), dtype=np.float32)

    # Matrice d'interaction R  (num_users × num_items)
    R = sp.csr_matrix((ones, (user_np, item_np)), shape=(num_users, num_items))

    # Matrice d'adjacence globale  A = [[0, R], [R.T, 0]]
    upper = sp.hstack([sp.csr_matrix((num_users, num_users)), R])
    lower = sp.hstack([R.T, sp.csr_matrix((num_items, num_items))])
    adj_mat = sp.vstack([upper, lower]).tocsr()

    # Normalisation symetrique  D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    rowsum[rowsum == 0] = 1e-7          # on evite la division par zéro

    d_inv_sqrt = np.power(rowsum, -0.5)
    d_mat      = sp.diags(d_inv_sqrt)
    norm_adj   = d_mat.dot(adj_mat).dot(d_mat).tocoo()

    # Conversion en tenseur COO
    indices = torch.from_numpy(
        np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
    )
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape  = torch.Size(norm_adj.shape)

    return torch.sparse_coo_tensor(indices, values, shape)