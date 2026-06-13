import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    Phase 3: Architecture LightGCN en PyTorch.

    LightGCN simplifie les GCN traditionnels en supprimant :
      - les transformations linéaires par couche (W_k)
      - les fonctions d'activation non linéaires
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_users     = num_users
        self.num_items     = num_items
        self.embedding_dim = embedding_dim
        self.num_layers    = num_layers

        # Embeddings de la couche 0 
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Xavier uniform, meilleure stabilite que N(0, 0.1) pour des
        # embeddings utilisees dans un produit scalaire
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, norm_adj: torch.Tensor):
        """
        Propagation multicouche sur le graphe biparti.
        """
        # concatenation utilisateurs + films
        ego = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )                                          # (N, d),  N = U + I

        all_embeddings = [ego]

        current = ego
        for _ in range(self.num_layers):
            current = torch.sparse.mm(norm_adj, current)
            all_embeddings.append(current)

        # moyenne sur les K+1 couches (poids uniformes 1/(K+1))
        final = torch.stack(all_embeddings, dim=0).mean(dim=0)   # (N, d)

        users_final, items_final = torch.split(
            final, [self.num_users, self.num_items], dim=0
        )
        return users_final, items_final

    def get_ego_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight