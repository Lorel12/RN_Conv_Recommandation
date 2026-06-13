import numpy as np
import torch

def sample_negative_pairwise(train_df, num_items):
    """
    Phase 4 : Échantillonnage négatif vectorisé (pairwise BPR).

    """
    user_np    = train_df['user_idx'].values
    pos_np     = train_df['item_idx'].values

    # Ensemble des paires positives pour le rejection sampling
    pos_set = set(zip(user_np.tolist(), pos_np.tolist()))

    # Tirage initial vectorisé
    neg_np  = np.random.randint(0, num_items, size=len(user_np))

    # Correction des collisions (en pratique très peu nombreuses)
    for i, (u, neg_i) in enumerate(zip(user_np, neg_np)):
        while (int(u), int(neg_i)) in pos_set:
            neg_i = np.random.randint(0, num_items)
        neg_np[i] = neg_i

    return (
        torch.tensor(user_np, dtype=torch.long),
        torch.tensor(pos_np, dtype=torch.long),
        torch.tensor(neg_np, dtype=torch.long),
    )

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking (BPR) loss """
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

def train_one_epoch(
    model,
    norm_adj,
    train_df,
    num_items: int,
    optimizer,
    lambda_reg: float = 1e-4,
    batch_size: int = 1024,
) -> float:
    """
    Phase 4 : Une époque d'entraînement avec la perte BPR + régularisation L2.
    """
    model.train()
    device = next(model.parameters()).device

    # Echantillonnage negatif (une fois par epoch) 
    users, pos_items, neg_items = sample_negative_pairwise(train_df, num_items)
    dataset_size = len(users)
    permutation  = torch.randperm(dataset_size)

    # Forward pass: propagation sur le graphe entier 
    users_final, items_final = model(norm_adj)   # (num_users, d), (num_items, d)

    epoch_loss = 0.0

    for i in range(0, dataset_size, batch_size):
        idx = permutation[i : i + batch_size]

        b_users = users[idx].to(device)
        b_pos = pos_items[idx].to(device)
        b_neg = neg_items[idx].to(device)

        # Embeddings finaux (propagés) pour ce batch
        u_emb = users_final[b_users]
        pos_emb = items_final[b_pos]
        neg_emb = items_final[b_neg]

        # Scores par produit scalaire
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        loss_bpr = bpr_loss(pos_scores, neg_scores)

        # Régularisation L2 sur les embeddings de la couche 0 uniquement
        u_0 = model.user_embedding(b_users)
        pos_0 = model.item_embedding(b_pos)
        neg_0 = model.item_embedding(b_neg)

        reg_loss = lambda_reg * (
            u_0.norm(2).pow(2) + pos_0.norm(2).pow(2) + neg_0.norm(2).pow(2)
        ) / len(b_users)

        loss = loss_bpr + reg_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  
        optimizer.step()                  

        epoch_loss += loss.item() * len(b_users)

    return epoch_loss / dataset_size