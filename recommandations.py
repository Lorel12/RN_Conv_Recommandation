import numpy as np

def generate_recommendations(user_idx, users_embeddings, items_embeddings, train_df, movies_df, top_k= 20):

    user_emb = users_embeddings[user_idx]
    items_emb = items_embeddings

    scores = user_emb @ items_emb.T

    seen_items = train_df[train_df["user_idx"] == user_idx]["item_idx"].values
    scores[seen_items] = -np.inf

    top_items = np.argsort(scores)[::-1][:top_k]

    item_to_movie = (
        train_df[["item_idx","movieId"]]
        .drop_duplicates()
        .set_index("item_idx")["movieId"]
        .to_dict()
    )

    movie_lookup = movies_df.set_index("movieId")

    results = []

    max_score = scores[top_items[0]] if len(top_items) > 0 else 1e-8

    for item_idx in top_items:

        movie_id = item_to_movie[item_idx]
        movie = movie_lookup.loc[movie_id]

        match_pct = (scores[item_idx] / max_score) * 100

        results.append({
            "movieId": int(movie_id),
            "title": movie["title"],
            "genres": movie["genres"].split("|"),
            "score": round(float(match_pct),1)
        })

    return results