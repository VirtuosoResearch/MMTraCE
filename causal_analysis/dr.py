import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--state_name', type=str, default="DE")
parser.add_argument("--neighbor", type=int, default=1)
args = parser.parse_args()
states = ["DE", "MA", "MD", "NV", "MT", "IA"]

for state in states:
    df_edge = pd.read_csv(f"./data/{state}/Road_Network_Edges_{state}.csv")
    df_acc = pd.read_csv(f"./data/{state}/accidents_monthly.csv")

    df_edge["is_motorway"] = (df_edge["highway"] == "motorway").astype(int)

    df_node = pd.read_csv(f"./data/{state}/Road_Network_Nodes_{state}.csv")
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(df_node["node_id"])}

    embed_dir = f"./output_emb/{state}_final"
    results = []
    psm_atts = []
    ipw_atts = []
    dr_ates = []

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
            except Exception:
                continue

            x_i = np.concatenate([h1, h2])
            t_i = row.is_motorway
            y_i = 1 if key in accident_set else 0

            X.append(x_i)
            T.append(t_i)
            Y.append(y_i)

        if len(X) == 0:
            continue

        X = np.array(X)
        T = np.array(T)
        Y = np.array(Y)

        treat_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        if len(treat_idx) == 0 or len(control_idx) == 0:
            continue

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        ps_model = LogisticRegression(max_iter=300, solver='lbfgs')
        ps_model.fit(Xs, T)
        e = ps_model.predict_proba(Xs)[:, 1]
        e = np.clip(e, 1e-6, 1 - 1e-6)

        nn = NearestNeighbors(n_neighbors=args.neighbor).fit(e[control_idx].reshape(-1, 1))
        dists, idxs = nn.kneighbors(e[treat_idx].reshape(-1, 1))
        matched_y = Y[control_idx][idxs[:, 0]]
        att_psm = (Y[treat_idx] - matched_y).mean()
        psm_atts.append(att_psm)

        w_ctrl = e[control_idx] / (1.0 - e[control_idx])
        att_ipw = Y[treat_idx].mean() - np.average(Y[control_idx], weights=w_ctrl)
        ipw_atts.append(att_ipw)

        m1 = LogisticRegression(max_iter=300, solver='lbfgs')
        m0 = LogisticRegression(max_iter=300, solver='lbfgs')
        m1.fit(Xs[treat_idx], Y[treat_idx])
        m0.fit(Xs[control_idx], Y[control_idx])

        m1x = m1.predict_proba(Xs)[:, 1]
        m0x = m0.predict_proba(Xs)[:, 1]

        psi = (m1x - m0x) + T * (Y - m1x) / e - (1 - T) * (Y - m0x) / (1 - e)
        tau_dr_ate = psi.mean()
        dr_ates.append(tau_dr_ate)

        results.append({
            "year": year,
            "month": month,
            "ATT_motorway_PSM": att_psm,
            "ATT_motorway_IPW": att_ipw,
            "ATE_motorway_DR": tau_dr_ate,
            "treated_count": len(treat_idx),
            "control_count": len(control_idx)
        })

    att_df = pd.DataFrame(results).sort_values(["year", "month"])
    print(f"\n===== {state} =====")
    print(att_df)

    def safe_mean_std(v):
        v = [x for x in v if x is not None and not np.isnan(x)]
        if len(v) == 0:
            return None, None
        return float(np.mean(v)), float(np.std(v))

    m_psm, s_psm = safe_mean_std(psm_atts)
    m_ipw, s_ipw = safe_mean_std(ipw_atts)
    m_dr,  s_dr  = safe_mean_std(dr_ates)

    print(f"PSM-ATT mean: {m_psm}, std: {s_psm}")
    print(f"IPW-ATT mean: {m_ipw}, std: {s_ipw}")
    print(f"DR-ATE  mean: {m_dr},  std: {s_dr}")
