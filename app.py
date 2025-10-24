import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock 2024 — Réceptions", layout="wide", page_icon="📦")
st.title("📦 Analyse des réceptions 2024 — Fichier unique")

st.markdown(
    "Importez votre fichier (CSV/Excel) avec les colonnes: "
    "`id`, `date_receptic`, `poids_brute`, `zone`, `num_eta`, `num_extc`."
)

# ---------------- Sidebar: upload & settings ----------------
with st.sidebar:
    st.header("Fichier de données")
    up = st.file_uploader("CSV ou Excel", type=["csv", "xlsx", "xls"])
    st.divider()
    st.header("Paramètres")
    year = st.number_input("Année à analyser", min_value=2000, max_value=2100, value=2024, step=1)
    csv_sep = st.text_input("Séparateur CSV", value=",")
    csv_decimal = st.text_input("Décimal CSV", value=".")
    csv_encoding = st.text_input("Encodage CSV", value="utf-8")
    forecast_weeks = st.slider("Horizon prévision (semaines)", 4, 16, 8, 1)
    do_anomaly = st.checkbox("Détection d'anomalies de poids", value=True)

def load_df(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file, sep=csv_sep, decimal=csv_decimal, encoding=csv_encoding)
        except Exception:
            # essai fallback
            df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

df_raw = load_df(up)
if df_raw is None:
    st.info("Veuillez importer votre fichier.")
    st.stop()

st.success(f"Fichier chargé: {up.name} | {df_raw.shape[0]:,} lignes, {df_raw.shape[1]:,} colonnes")

# ---------------- Column guessing ----------------
cols = [c for c in df_raw.columns]

def guess(colnames, candidates):
    low = {c.lower(): c for c in colnames}
    for c in candidates:
        if c in low:
            return low[c]
    return None

date_col = guess(cols, ["date_receptic", "date_reception", "date"])
qty_col = guess(cols, ["poids_brute", "poids", "quantity", "qty"])
zone_col = guess(cols, ["zone", "site", "entrepot", "warehouse"])
eta_col = guess(cols, ["num_eta", "eta", "numero_eta"])
extc_col = guess(cols, ["num_extc", "extc", "numero_extc"])

with st.expander("Vérifier/ajuster le mapping des colonnes"):
    date_col = st.selectbox("Colonne date", options=cols, index=cols.index(date_col) if date_col in cols else 0)
    qty_col = st.selectbox("Colonne poids (quantité)", options=cols, index=cols.index(qty_col) if qty_col in cols else 0)
    zone_col = st.selectbox("Colonne zone", options=[None] + cols, index=(cols.index(zone_col)+1) if zone_col in cols else 0)
    eta_col = st.selectbox("Colonne numéro ETA", options=[None] + cols, index=(cols.index(eta_col)+1) if eta_col in cols else 0)
    extc_col = st.selectbox("Colonne numéro EXTC", options=[None] + cols, index=(cols.index(extc_col)+1) if extc_col in cols else 0)

# ---------------- Clean & filter 2024 ----------------
df = df_raw.copy()

# dates
df["__date"] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=["__date"])

# quantité: gérer virgule décimale si lue comme texte
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")

df["__qty"] = df[qty_col].apply(to_num)
df["__qty"] = df["__qty"].fillna(0.0)

df["__zone"] = df[zone_col].astype(str) if zone_col else "UNSPECIFIED"
df["__eta"] = df[eta_col].astype(str) if eta_col else ""
df["__extc"] = df[extc_col].astype(str) if extc_col else ""

# Filtrer sur l'année
df["__year"] = df["__date"].dt.year
df = df[df["__year"] == int(year)].copy()

if df.empty:
    st.warning(f"Aucune ligne pour l'année {year}.")
    st.stop()

# ---------------- KPIs ----------------
min_d, max_d = df["__date"].min().date(), df["__date"].max().date()
total_qty = df["__qty"].sum()
nb_rows = df.shape[0]
nb_zones = df["__zone"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Volume total {year}", f"{total_qty:,.1f}")
c2.metric("Nb réceptions", f"{nb_rows:,}")
c3.metric("Nb zones", f"{nb_zones:,}")
c4.metric("Période", f"{min_d} → {max_d}")

# ---------------- Aggregations ----------------
daily = (
    df.groupby("__date", as_index=False)
      .agg(qty=("__qty","sum"))
      .sort_values("__date")
)

by_zone = (
    df.groupby("__zone", as_index=False)
      .agg(qty=("__qty","sum"), receptions=("__qty","count"))
      .sort_values("qty", ascending=False)
)

df["month"] = df["__date"].dt.month
zone_month = (
    df.groupby(["__zone","month"], as_index=False)
      .agg(qty=("__qty","sum"))
)

# ---------------- Charts ----------------
st.subheader("Série temporelle quotidienne (poids)")
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=daily["__date"], y=daily["qty"], mode="lines", name="Poids"))
fig_ts.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig_ts, use_container_width=True)

st.subheader("Top zones par volume")
topn = min(15, by_zone.shape[0])
fig_zone = px.bar(by_zone.head(topn), x="__zone", y="qty", text_auto=".2s", labels={"__zone":"Zone","qty":"Poids"})
fig_zone.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=10), xaxis_title="Zone", yaxis_title="Poids")
st.plotly_chart(fig_zone, use_container_width=True)

st.subheader("Heatmap Zone × Mois (poids)")
pivot = zone_month.pivot(index="__zone", columns="month", values="qty").fillna(0)
fig_hm = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Poids"))
st.plotly_chart(fig_hm, use_container_width=True)

# ---------------- Data quality checks ----------------
st.subheader("Contrôles qualité")
issues = []

if eta_col:
    dup_eta = df[df["__eta"].duplicated(keep=False)].sort_values("__eta")
    if not dup_eta.empty:
        issues.append(f"Doublons num_eta: {dup_eta['__eta'].nunique()} valeurs")
        with st.expander("Voir doublons num_eta"):
            st.dataframe(dup_eta[[date_col, eta_col, qty_col, zone_col]])
if extc_col:
    dup_extc = df[df["__extc"].duplicated(keep=False)].sort_values("__extc")
    if not dup_extc.empty:
        issues.append(f"Doublons num_extc: {dup_extc['__extc'].nunique()} valeurs")
        with st.expander("Voir doublons num_extc"):
            st.dataframe(dup_extc[[date_col, extc_col, qty_col, zone_col]])

null_qty = df[df["__qty"].isna() | (df["__qty"] <= 0)]
if not null_qty.empty:
    issues.append(f"Lignes à quantité nulle/négative: {null_qty.shape[0]}")

if issues:
    st.warning(" | ".join(issues))
else:
    st.success("Contrôles OK (aucun doublon critique ou quantité nulle).")

# ---------------- Anomaly detection ----------------
st.subheader("Anomalies de poids (réceptions individuelles)")
if do_anomaly:
    # Anomalies sur la distribution des poids individuels
    series = df[["__qty"]].copy()
    if len(series) >= 30:
        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(series.values)
        df["__anomaly"] = (labels == -1)
        anom = df[df["__anomaly"]]
        fig_a = px.scatter(df, x="__date", y="__qty", color="__anomaly",
                           color_discrete_map={False:"#1f77b4", True:"#d62728"},
                           labels={"__date":"Date","__qty":"Poids","__anomaly":"Anomalie"},
                           title="Points anormaux sur le poids des réceptions")
        st.plotly_chart(fig_a, use_container_width=True)
        with st.expander("Table des anomalies"):
            st.dataframe(anom[[date_col, qty_col, zone_col, eta_col, extc_col]].sort_values(date_col))
    else:
        st.info("Série trop courte (<30) pour la détection d'anomalies.")

# ---------------- Forecast ----------------
st.subheader("Prévision du volume de réceptions")
# agrégation journalière (remplir jours manquants à 0)
ds = daily.set_index("__date").asfreq("D").fillna(0.0)
if len(ds) >= 30 and ds["qty"].sum() > 0:
    try:
        model = auto_arima(ds["qty"], seasonal=False, stepwise=True, suppress_warnings=True)
        horizon_days = int(forecast_weeks * 7)
        fc = model.predict(n_periods=horizon_days)
        idx_future = pd.date_range(ds.index[-1] + timedelta(days=1), periods=horizon_days, freq="D")
        fc_df = pd.DataFrame({"date": idx_future, "forecast_qty": fc})
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=ds.index, y=ds["qty"], mode="lines", name="Historique"))
        fig_fc.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast_qty"], mode="lines", name="Prévision"))
        fig_fc.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_fc, use_container_width=True)
        st.download_button("📥 Télécharger prévision (CSV)", fc_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_receptions_{year}.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"Échec de la prévision: {e}")
else:
    st.info("Série journalière insuffisante pour une prévision fiable (≥30 jours non nuls recommandé).")

# ---------------- Exports ----------------
st.subheader("Exports")
agg_month = (
    df.assign(month=lambda d: d["__date"].dt.to_period("M").astype(str))
      .groupby(["month","__zone"], as_index=False)["__qty"].sum()
      .rename(columns={"__zone":"zone","__qty":"poids"})
)
st.download_button("📥 Export mensuel par zone (CSV)",
                   agg_month.to_csv(index=False).encode("utf-8"),
                   file_name=f"mensuel_zone_{year}.csv", mime="text/csv")

st.download_button("📥 Export par réception (CSV)",
                   df[[date_col, qty_col, zone_col, eta_col, extc_col]].to_csv(index=False).encode("utf-8"),
                   file_name=f"receptions_{year}.csv", mime="text/csv")