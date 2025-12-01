import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--state_name', type=str, default="DE")
args = parser.parse_args()
states = ["DE", "MA", "MD", "NV", "MT", "IA"]

for state in states:
    df_edge = pd.read_csv(f"./data/{state}/Road_Network_Edges_{state}.csv")
    df_acc = pd.read_csv(f"./data/{state}/accidents_monthly.csv")

    df_edge["is_motorway"] = (df_edge["highway"] == "motorway").astype(int)
    edge2treatment = {(r.node_1, r.node_2): r.is_motorway for r in df_edge.itertuples()}

    df_node = pd.read_csv(f"./data/{state}/Road_Network_Nodes_{state}.csv")
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(df_node["node_id"])}

    embed_dir = f"./output_emb/{state}_final"
    results = []
    all_att = []

    for fname in tqdm(sorted(os.listdir(embed_dir)), total=len(os.listdir(embed_dir))):
        if not fname.endswith(".npy"):
            continue

        year, month = map(int, fname.replace(".npy", "").split("_"))
        embed = np.load(os.path.join(embed_dir, fname))

        df_month = df_acc[(df_acc["year"] == year) & (df_acc["month"] == month)]
        accident_set = set(zip(df_month.node_1, df_month.node_2)) | set(zip(df_month.node_2, df_month.node_1))

        X, T, Y = [], [], []
        for row in df_edge.itertuples():
            key = (row.node_1, row.node_2)

            try:
                idx1 = node_id_to_idx[row.node_1]
                idx2 = node_id_to_idx[row.node_2]
                h1 = embed[idx1]
                h2 = embed[idx2]
            except:
                continue

            x_i = np.concatenate([h1, h2])
            t_i = row.is_motorway
            y_i = 1 if key in accident_set else 0

            X.append(x_i)
            T.append(t_i)
            Y.append(y_i)

        X = np.array(X)
        T = np.array(T)
        Y = np.array(Y)

        treat_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        if len(treat_idx) == 0 or len(control_idx) == 0:
            continue

        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, T)
        prop_scores = ps_model.predict_proba(X)[:, 1]  # P(T=1|X)

        nn = NearestNeighbors(n_neighbors=1).fit(prop_scores[control_idx].reshape(-1, 1))
        dists, idxs = nn.kneighbors(prop_scores[treat_idx].reshape(-1, 1))
        matched_y = Y[control_idx][idxs[:, 0]]

        att = (Y[treat_idx] - matched_y).mean()
        if att > 0.1:
            all_att.append(att)

        results.append({
            "year": year,
            "month": month,
            "ATT_motorway": att,
            "treated_count": len(treat_idx),
            "control_count": len(control_idx)
        })

    att_df = pd.DataFrame(results).sort_values(["year", "month"])
    print(f"\n===== {state} =====")
    print(att_df)
    print("mean: ", np.mean(all_att), "std: ", np.std(all_att))
