import warnings
from datetime import timedelta
import re
import unicodedata
from io import BytesIO
import plotly.express as px                                           
import plotly.graph_objects as go


import pandas as pd
import streamlit as st

# Imports facultatifs (protégés) pour éviter un crash si non installés
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

try:
    from sklearn.ensemble import IsolationForest
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    from pmdarima import auto_arima
    HAVE_ARIMA = True
except Exception:
    HAVE_ARIMA = False

warnings.filterwarnings("ignore")

# ---------------- Mise en page ----------------
APP_TITLE = "Analyse des réceptions 2024 — Fichier unique"
st.set_page_config(page_title="Stock 2024 — Réceptions", layout="wide", page_icon="📦")
st.title(f"📦 {APP_TITLE}")
st.markdown(
    "Importez votre fichier (CSV/Excel) avec les colonnes: "
    "`id`, `date_receptic`, `poids_brute`, `zone`, `num_eta`, `num_extc`."
)

# ---------------- Fonctions utilitaires ----------------
def normalize_zone(z: str, merge_k_variants: bool = True) -> str:
    """Nettoyage/normalisation robuste des libellés de zone (espaces/tirets/Unicode)."""
    if pd.isna(z):
        return "UNSPECIFIED"
    s = str(z)
    s = unicodedata.normalize("NFKC", s)  # normalise chiffres/lettres exotiques (ex: ١١١ → 111)
    # Remplacer tirets exotiques par "-"
    s = (s.replace("\u2010","-")  # hyphen
           .replace("\u2011","-") # non-breaking hyphen
           .replace("\u2012","-").replace("\u2013","-").replace("\u2014","-").replace("\u2212","-"))
    # Retirer espaces invisibles
    s = (s.replace("\u00A0"," ")  # NBSP
           .replace("\u202F"," ")
           .replace("\u2009"," ")
           .replace("\u200A"," ")
           .replace("\u200B",""))
    s = s.strip().upper()
    if merge_k_variants:
        s2 = re.sub(r"[\s_]+", "-", s)              # espaces/underscore → "-"
        s2 = re.sub(r"^K[-\s_]*111$", "K-111", s2)  # unifie K111/K 111/K_111 → K-111
        return s2
    return s

@st.cache_data(show_spinner=False)
def read_csv_bytes(b: bytes, sep: str = ",", decimal: str = ".", encoding: str = "utf-8") -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(b), sep=sep, decimal=decimal, encoding=encoding)
    except Exception:
        return pd.read_csv(BytesIO(b))

@st.cache_data(show_spinner=False)
def read_excel_bytes(b: bytes, sheet_name=None) -> pd.DataFrame:
    return pd.read_excel(BytesIO(b), sheet_name=sheet_name)

def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")

def codepoints(s: str) -> str:
    s = str(s)
    return " ".join(hex(ord(c)) for c in s)

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
    contamination_pct = st.slider("Taux d'anomalies (%)", min_value=1, max_value=10, value=5, step=1)
    normalize_zones = st.checkbox("Normaliser libellés de zones", value=True)

# Gestion absence de Plotly (évite l'erreur opaque de Streamlit Cloud)
if not HAVE_PLOTLY:
    st.error(
        "Plotly n'est pas installé dans l'environnement. "
        "Ajoutez 'plotly' dans requirements.txt puis redéployez."
    )
    st.stop()

# ---------------- Chargement fichier ----------------
if up is None:
    st.info("Veuillez importer votre fichier.")
    st.stop()

bytes_data = up.getvalue()
name = up.name.lower()

df_raw = None
sheet_choice = None
if name.endswith(".csv"):
    df_raw = read_csv_bytes(bytes_data, sep=csv_sep, decimal=csv_decimal, encoding=csv_encoding)
else:
    # Permettre le choix de la feuille si Excel
    with pd.ExcelFile(BytesIO(bytes_data)) as xls:
        sheets = xls.sheet_names
    if len(sheets) > 1:
        sheet_choice = st.sidebar.selectbox("Feuille Excel", sheets, index=0)
    else:
        sheet_choice = sheets[0]
    df_raw = read_excel_bytes(bytes_data, sheet_name=sheet_choice)

st.success(
    f"Fichier chargé: {up.name}"
    + (f" — feuille: {sheet_choice}" if sheet_choice else "")
    + f" | {df_raw.shape[0]:,} lignes, {df_raw.shape[1]:,} colonnes"
)

# ---------------- Mapping des colonnes ----------------
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

with st.expander("Vérifier/ajuster le mapping des colonnes", expanded=False):
    date_col = st.selectbox(
        "Colonne date", options=cols, index=cols.index(date_col) if date_col in cols else 0
    )
    qty_col = st.selectbox(
        "Colonne poids (quantité)", options=cols, index=cols.index(qty_col) if qty_col in cols else 0
    )
    zone_col = st.selectbox(
        "Colonne zone", options=[None] + cols, index=(cols.index(zone_col) + 1) if zone_col in cols else 0
    )
    eta_col = st.selectbox(
        "Colonne numéro ETA", options=[None] + cols, index=(cols.index(eta_col) + 1) if eta_col in cols else 0
    )
    extc_col = st.selectbox(
        "Colonne numéro EXTC", options=[None] + cols, index=(cols.index(extc_col) + 1) if extc_col in cols else 0
    )

# Contrôle minimum
if date_col is None or qty_col is None:
    st.error("La colonne date et la colonne poids sont obligatoires.")
    st.stop()

# ---------------- Nettoyage & filtre année ----------------
df = df_raw.copy()

# Date
df["__date"] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=["__date"])

# Quantité
df["__qty"] = df[qty_col].apply(to_num).fillna(0.0)

# Zone (brut + normalisée)
if zone_col:
    df["__zone_raw"] = df[zone_col].astype(str)
    df["__zone"] = df["__zone_raw"].apply(lambda z: normalize_zone(z, True) if normalize_zones else str(z))
else:
    df["__zone_raw"] = "UNSPECIFIED"
    df["__zone"] = "UNSPECIFIED"

# Identifiants
df["__eta"] = df[eta_col].astype(str) if eta_col else ""
df["__extc"] = df[extc_col].astype(str) if extc_col else ""

# Filtre année choisie
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

# ---------------- Agrégations ----------------
daily = (
    df.groupby("__date", as_index=False)
      .agg(qty=("__qty", "sum"))
      .sort_values("__date")
)

by_zone = (
    df.groupby("__zone", as_index=False)
      .agg(qty=("__qty", "sum"), receptions=("__qty", "count"))
      .sort_values("qty", ascending=False)
)

df["month"] = df["__date"].dt.month
zone_month = (
    df.groupby(["__zone", "month"], as_index=False)
      .agg(qty=("__qty", "sum"))
)

# ---------------- Graphiques ----------------
st.subheader("Série temporelle quotidienne (poids)")
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=daily["__date"], y=daily["qty"], mode="lines", name="Poids"))
fig_ts.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_ts, use_container_width=True)

st.subheader("Top zones par volume")
topn = min(15, by_zone.shape[0])
fig_zone = px.bar(
    by_zone.head(topn), x="__zone", y="qty",
    text_auto=".2s", labels={"__zone": "Zone", "qty": "Poids"}
)
fig_zone.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Zone", yaxis_title="Poids")
st.plotly_chart(fig_zone, use_container_width=True)

st.subheader("Heatmap Zone × Mois (poids)")
pivot = zone_month.pivot(index="__zone", columns="month", values="qty").fillna(0)
# Ordonner les mois
pivot = pivot.reindex(sorted(pivot.columns), axis=1)
fig_hm = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Poids"))
st.plotly_chart(fig_hm, use_container_width=True)

# Diagnostic zones (utile si K111 / K‑111, etc.)
with st.expander("Diagnostic zones (valeurs distinctes et caractères)"):
    tmp = (
        df.groupby(["__zone", "__zone_raw"], as_index=False)
          .agg(n=("__zone", "size"), poids=("__qty", "sum"))
          .sort_values("n", ascending=False)
          .head(50)
    )
    tmp["raw_codepoints"] = tmp["__zone_raw"].apply(codepoints)
    st.dataframe(tmp)

# ---------------- Contrôles qualité ----------------
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

# ---------------- Détection d'anomalies ----------------
st.subheader("Anomalies de poids (réceptions individuelles)")
if do_anomaly:
    if not HAVE_SKLEARN:
        st.info("Scikit-learn n'est pas installé: la détection d'anomalies est désactivée. "
                "Ajoutez 'scikit-learn' dans requirements.txt.")
    else:
        series = df[["__qty"]].copy()
        if len(series) >= 30:
            contam = max(0.01, min(0.2, contamination_pct / 100.0))
            iso = IsolationForest(contamination=contam, random_state=42)
            labels = iso.fit_predict(series.values)
            df["__anomaly"] = (labels == -1)
            anom = df[df["__anomaly"]]
            fig_a = px.scatter(
                df, x="__date", y="__qty", color="__anomaly",
                color_discrete_map={False: "#1f77b4", True: "#d62728"},
                labels={"__date": "Date", "__qty": "Poids", "__anomaly": "Anomalie"},
                title="Points anormaux sur le poids des réceptions"
            )
            st.plotly_chart(fig_a, use_container_width=True)
            with st.expander("Table des anomalies"):
                cols_to_show = [c for c in [date_col, qty_col, zone_col, eta_col, extc_col] if c]
                st.dataframe(anom[cols_to_show].sort_values(date_col))
        else:
            st.info("Série trop courte (<30) pour la détection d'anomalies.")

# ---------------- Prévision ----------------
st.subheader("Prévision du volume de réceptions")
# Agrégation journalière (remplir jours manquants à 0)
ds = daily.set_index("__date").asfreq("D").fillna(0.0)
if len(ds) >= 30 and ds["qty"].sum() > 0:
    if not HAVE_ARIMA:
        st.info("pmdarima n'est pas installé: la prévision est désactivée. "
                "Ajoutez 'pmdarima' et 'statsmodels' dans requirements.txt.")
    else:
        try:
            model = auto_arima(ds["qty"], seasonal=False, stepwise=True, suppress_warnings=True)
            horizon_days = int(forecast_weeks * 7)
            fc = model.predict(n_periods=horizon_days)
            idx_future = pd.date_range(ds.index[-1] + timedelta(days=1), periods=horizon_days, freq="D")
            fc_df = pd.DataFrame({"date": idx_future, "forecast_qty": fc})
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=ds.index, y=ds["qty"], mode="lines", name="Historique"))
            fig_fc.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast_qty"], mode="lines", name="Prévision"))
            fig_fc.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_fc, use_container_width=True)
            st.download_button(
                "📥 Télécharger prévision (CSV)",
                fc_df.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_receptions_{year}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.warning(f"Échec de la prévision: {e}")
else:
    st.info("Série journalière insuffisante pour une prévision fiable (≥30 jours non nuls recommandé).")

# ---------------- Exports ----------------
st.subheader("Exports")
agg_month = (
    df.assign(month=lambda d: d["__date"].dt.to_period("M").astype(str))
      .groupby(["month", "__zone"], as_index=False)["__qty"].sum()
      .rename(columns={"__zone": "zone", "__qty": "poids"})
)
st.download_button(
    "📥 Export mensuel par zone (CSV)",
    agg_month.to_csv(index=False).encode("utf-8"),
    file_name=f"mensuel_zone_{year}.csv", mime="text/csv"
)

cols_to_export = [c for c in [date_col, qty_col, zone_col, eta_col, extc_col] if c]
st.download_button(
    "📥 Export par réception (CSV)",
    df[cols_to_export].to_csv(index=False).encode("utf-8"),
    file_name=f"receptions_{year}.csv", mime="text/csv"
)



