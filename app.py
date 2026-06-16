import os, json, numpy as np, pandas as pd, pickle
from flask import Flask, render_template, request
from recommandations import generate_recommendations

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_PATH = os.path.join(BASE_DIR, "lightgcn_artifacts.pkl")

with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)

users_embeddings = artifacts["users_embeddings"]
items_embeddings = artifacts["items_embeddings"]
user_mapping = artifacts["user_mapping"]
item_mapping = artifacts["item_mapping"]
movies_df = artifacts["movies_df"]
train_df = artifacts["train_df"]

final_recall = artifacts.get("recall", 0)
final_ndcg = artifacts.get("ndcg", 0)
final_loss = artifacts.get("loss", 0)

user_to_idx = user_mapping
idx_to_item = {v: k for k, v in item_mapping.items()}

users_list = sorted(user_to_idx.keys())


@app.route("/")
def interface_client():

    selected_user = request.args.get(
        "user_id",
        default=users_list[0],
        type=int
    )

    calculer = request.args.get("calculer") == "true"

    user_idx = user_to_idx[selected_user]

    # historique simplifié (propre + rapide)
    history_ids = train_df[train_df["user_idx"] == user_idx]["movieId"].unique()

    historique = movies_df[
        movies_df["movieId"].isin(history_ids)
    ].head(6).to_dict(orient="records")

    recommandations = []

    if calculer:
        recommandations = generate_recommendations(
            user_idx,
            users_embeddings,
            items_embeddings,
            train_df,
            movies_df,
            top_k=20
        )

    return render_template(
        "index.html",
        users=users_list[:100],
        selected_user=selected_user,
        historique=historique,
        recommandations=recommandations,
        en_attente=not calculer
    )


@app.route("/admin")
def interface_admin():

    metrics = {
        "users": len(user_mapping),
        "movies": len(item_mapping),
        "embedding_dim": users_embeddings.shape[1],
        "recall": f"{final_recall*100:.2f} %",
        "ndcg": f"{final_ndcg*100:.2f} %",
        "loss_finale": f"{final_loss:.4f}"
    }

    perf_path = os.path.join(BASE_DIR, "model_metrics.json")

    if os.path.exists(perf_path):
        with open(perf_path, "r") as f:
            metrics.update(json.load(f))

    return render_template("admin.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True, port=5000)