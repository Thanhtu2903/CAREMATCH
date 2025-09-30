
# -*- coding: utf-8 -*-
"""
Carematch Dashboard
Clean Streamlit version
"""

# === Imports ===
import yake
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans       # works with sparse
from sklearn.decomposition import TruncatedSVD
# === Load Dataset ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
carematch = pd.read_csv("carematch_requests.csv")
st.markdown(""" ***GROUP 4***: TU PHAM & MINH NGUYEN""")
# === Dashboard Title ===
st.title("📊 Carematch Dashboard")

# === Introduction / Project Background ===
st.header("🏥 Project Background")
st.markdown("""**CareMatch Health** is a regional healthcare network serving a diverse patient population across both urban and suburban communities.  
Patients submit appointment requests and complete intake forms through the organization’s digital platforms.  

Although CareMatch holds a large volume of patient and operational data, it has not yet implemented advanced analytics or AI-powered tools to derive value from this information.
➡️ As a result, the immediate need is to **explore the data, identify opportunities, extract actionable insights, and build data-driven solutions** that can improve access, efficiency, and patient experience.
""")
# === Show Sample Data ===
st.subheader("Sample Data")
st.write(carematch.head())

# === Descriptive Stats ===
st.header("📊 Descriptive Statistics (All Variables)")
desc_stats = carematch.describe(include="all").T
st.dataframe(desc_stats)

# === Histogram Plots ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("Wait Time Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(carematch['wait_time'], bins=20, kde=False, color='blue', ax=ax1)
    st.pyplot(fig1)
st.markdown(""" Wait time are spread out without a strong concentration at a particular interval""")
with col2:
    st.subheader("Chronic Conditions Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(carematch['chronic_conditions_count'], bins=20, kde=False, color='blue', ax=ax2)
    st.pyplot(fig2)
st.markdown(""" Most patients present with 0–2 chronic conditions, with 1 chronic condition being the most common.
This distribution highlights that while the majority of cases are relatively simple, resource planning should account for a smaller group of patients with complex healthcare needs.""")
# === Boxplots ===
st.header("📊 Wait Time by Categories")
# --- Conclusion for Wait Time Analysis ---
st.markdown("""
### ✅ Conclusion: Wait Time Analysis

- Wait times are fairly consistent across **language preference, provider specialty, and urgency score**.  
- The **median wait time is ~15 days** for all groups, with wide variability.  
- This suggests that **individual patient characteristics and provider type do not strongly impact wait times**.  
- Instead, delays may be driven more by **system-level factors** such as scheduling efficiency and resource allocation.  
- ⚠️ Notably, **urgency score does not significantly reduce wait times**, highlighting a **misalignment between clinical need and scheduling practices**.
""")
st.subheader("Wait Time by Language Preference")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.boxplot(data=carematch, x="language_pref", y="wait_time", palette="Set3", ax=ax3)
st.pyplot(fig3)

st.subheader("Wait Time by Provider Specialty")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.boxplot(data=carematch, x="provider_specialty", y="wait_time", palette="Set3", ax=ax4)
st.pyplot(fig4)

st.subheader("Wait Time by Urgency Score")
fig5, ax5 = plt.subplots(figsize=(10,6))
sns.boxplot(data=carematch, x="urgency_score", y="wait_time", palette="Set3", ax=ax5)
st.pyplot(fig5)

# === Countplots ===
st.header("📊 Distribution of Categorical Variables")
col3, col4 = st.columns(2)
st.markdown("""**Urgency Score Distribution** is fairly balanced across all five levels, indicating that patients are being assigned urgency ratings in a relatively even manner. 
**Mental Health Flag** shows a strong imbalance: the vast majority of requests (~85%) are **not flagged for mental health**, while only a small fraction (~15%) are.""")
with col3:
    st.subheader("Urgency Score Distribution")
    fig6, ax6 = plt.subplots(figsize=(8,5))
    sns.countplot(data=carematch, x="urgency_score", order=carematch['urgency_score'].value_counts().index, ax=ax6)
    st.pyplot(fig6)

with col4:
    st.subheader("Mental Health Flag Distribution")
    fig7, ax7 = plt.subplots(figsize=(8,5))
    sns.countplot(data=carematch, x="mental_health_flag", order=carematch['mental_health_flag'].value_counts().index, ax=ax7)
    st.pyplot(fig7)

# === Word Cloud ===
st.header("☁️ Word Cloud of Condition Summaries")
st.markdown("""The word cloud provides a **quick thematic snapshot** of what patients are most frequently seeking help for, guiding providers on where to focus resources.""")
def preprocess(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

carematch['clean_summary'] = carematch['condition_summary'].apply(preprocess)
text = " ".join(carematch['clean_summary'])
stopwords = set(STOPWORDS)
stopwords.update(["need","ongoing","consultation","requesting","follow","patient"])

wordcloud = WordCloud(width=1200, height=600, background_color="white",
                      stopwords=stopwords, colormap="tab10", collocations=True).generate(text)

fig8, ax8 = plt.subplots(figsize=(12,6))
ax8.imshow(wordcloud, interpolation="bilinear")
ax8.axis("off")
st.pyplot(fig8)

# === Case & Provider Counts with Filters ===
st.header("📊 Case & Provider Counts with Filters")
st.sidebar.header("🔎 Filters")
st.markdown(""" 
- ***Provider Coverage by Location:** How many unique providers are available within each zip code?

- ***Workload Distribution by Month:*** How many patient cases are assigned to each provider on a monthly basis?

- ***Provider Case Volume:*** How many total cases each provider ID is responsible for managing, reflecting workload intensity.""")

zip_options = sorted(carematch['zip_code'].dropna().unique())
provider_options = sorted(carematch['assigned_provider_id'].dropna().unique())
selected_zip = st.sidebar.selectbox("Select a Zip Code", ["All"] + list(zip_options))
selected_provider = st.sidebar.selectbox("Select a Provider ID", ["All"] + list(provider_options))

# Cases per zip
cases_per_zip = carematch['zip_code'].value_counts().reset_index()
cases_per_zip.columns = ['zip_code', 'total_cases']
providers_per_zip = carematch.groupby("zip_code")["assigned_provider_id"].nunique().reset_index(name="unique_providers")
zip_summary = pd.merge(cases_per_zip, providers_per_zip, on="zip_code")
if selected_zip != "All":
    zip_summary = zip_summary[zip_summary['zip_code'] == selected_zip]
st.subheader("📍 Zip Code Summary")
st.dataframe(zip_summary)

# Provider case counts
provider_case_counts = carematch['assigned_provider_id'].value_counts().reset_index()
provider_case_counts.columns = ['assigned_provider_id', 'total_cases_for_provider']
if selected_provider != "All":
    provider_case_counts = provider_case_counts[provider_case_counts['assigned_provider_id'] == selected_provider]
st.subheader("👨‍⚕️ Provider Case Counts")
st.dataframe(provider_case_counts)

# Cases per provider within zip
zip_provider_cases = carematch.groupby(["zip_code", "assigned_provider_id"]).size().reset_index(name="case_count")
if selected_zip != "All":
    zip_provider_cases = zip_provider_cases[zip_provider_cases['zip_code'] == selected_zip]
if selected_provider != "All":
    zip_provider_cases = zip_provider_cases[zip_provider_cases['assigned_provider_id'] == selected_provider]
st.subheader("📍+👨‍⚕️ Cases per Provider within each Zip Code")
st.dataframe(zip_provider_cases)

# === Monthly Case Counts ===
st.header("📅 Monthly Case Counts per Provider")
carematch['request_timestamp'] = pd.to_datetime(carematch['request_timestamp'])
carematch['request_month'] = carematch['request_timestamp'].dt.to_period("M")

monthly_counts = carematch.groupby(['assigned_provider_id','request_month']).size().reset_index(name='case_count')
years = sorted(carematch['request_timestamp'].dt.year.unique())
months = sorted(carematch['request_timestamp'].dt.month.unique())
selected_year = st.sidebar.selectbox("Select Year", ["All"] + list(years))
selected_month = st.sidebar.selectbox("Select Month", ["All"] + list(months))

filtered = monthly_counts.copy()
if selected_year != "All":
    filtered = filtered[filtered['request_month'].dt.year == int(selected_year)]
if selected_month != "All":
    filtered = filtered[filtered['request_month'].dt.month == int(selected_month)]
st.subheader("📊 Case Counts per Provider (Filtered by Month/Year)")
st.dataframe(filtered)

# === Keyword Extraction ===
st.markdown("""***Data Preprocessing***:  
The dataset contained free-text entries under the column **condition_summary**.  
We extract a concise diagnosis keyword from each summary using **YAKE** to standardize inputs for clustering.""")

st.header("🩺 Keyword Extraction from Condition Summaries")

kw_extractor = yake.KeywordExtractor(top=1, stopwords=None)

def extract_keyword(text):
    if pd.isnull(text) or not str(text).strip():
        return None
    keywords = kw_extractor.extract_keywords(str(text))
    return keywords[0][0] if keywords else None

if "diagnosis" not in carematch.columns:
    carematch["diagnosis"] = carematch["condition_summary"].apply(extract_keyword)

st.subheader("Sample Condition Summaries with Diagnosis Keyword")
st.dataframe(carematch[["condition_summary","diagnosis"]].head(50))

keyword_counts = (carematch['diagnosis']
                  .dropna()
                  .value_counts()
                  .reset_index())
keyword_counts.columns = ["diagnosis_keyword","count"]

st.subheader("Most Frequent Diagnosis Keywords")
st.dataframe(keyword_counts.head(20))

# Bar plot of top keywords
fig9, ax9 = plt.subplots(figsize=(10,6))
sns.barplot(data=keyword_counts.head(15), x="count", y="diagnosis_keyword", ax=ax9)
ax9.set_xlabel("Count"); ax9.set_ylabel("Diagnosis keyword")
st.pyplot(fig9)

# ====================
# === Clustering  ====
# ====================
st.header("🤖 Patient Clustering Analysis")
st.markdown("""***Method:*** We combine TF-IDF vectors of the extracted **diagnosis** with three structured signals (**urgency_score**, **chronic_conditions_count**, **mental_health_flag**).  
We then run **MiniBatchKMeans** (works with sparse matrices) and visualize clusters with **TruncatedSVD** (PCA-like for sparse).""")

# Keep only rows that have a diagnosis keyword
mask = carematch["diagnosis"].notnull()
if mask.sum() < 5:
    st.warning("Not enough rows with extracted diagnosis to run clustering (need at least 5).")
else:
    try:
        # ---- Vectorize diagnosis (sparse) ----
        vectorizer = TfidfVectorizer(stop_words="english")
        X_text = vectorizer.fit_transform(carematch.loc[mask, "diagnosis"].astype(str))

        # ---- Scale structured features (dense to start, that’s fine) ----
        cluster_scaler = StandardScaler()
        X_struct = cluster_scaler.fit_transform(
            carematch.loc[mask, ["urgency_score","chronic_conditions_count","mental_health_flag"]]
        )

        # ---- Fuse into one sparse-like design (X_text is sparse; hstack keeps it efficient) ----
        X_cluster = hstack([X_text, X_struct])

        # ---- Elbow method to choose k ----
        st.header("📉 Elbow Method for Optimal k")
        inertia = []
        K = range(2, 11)
        for k_opt in K:
            kmb = MiniBatchKMeans(n_clusters=k_opt, random_state=42, n_init=10, batch_size=2048)
            kmb.fit(X_cluster)
            inertia.append(kmb.inertia_)
        fig10, ax10 = plt.subplots(figsize=(8,6))
        ax10.plot(list(K), inertia, "bo-")
        ax10.set_xlabel("Number of clusters (k)")
        ax10.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
        st.pyplot(fig10)

        # ---- Sidebar: choose k and fit final model ----
        st.sidebar.subheader("⚙️ Clustering Parameters")
        k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=2048)
        labels = kmeans.fit_predict(X_cluster)
        carematch.loc[mask, "cluster"] = labels

        # ---- 2D visualization with TruncatedSVD (PCA-like for sparse) ----
        st.subheader("📊 2D Visualization of Clusters")
        svd = TruncatedSVD(n_components=2, random_state=42)
        X_2d = svd.fit_transform(X_text)  # use text-only for a clean visual; could also use X_cluster
        fig11, ax11 = plt.subplots(figsize=(8,6))
        sns.scatterplot(
            x=X_2d[:,0], y=X_2d[:,1],
            hue=carematch.loc[mask, "cluster"].astype(int),
            palette="tab10", ax=ax11, legend=True
        )
        ax11.set_xlabel("Component 1"); ax11.set_ylabel("Component 2")
        st.pyplot(fig11)

        # ---- Cluster insights ----
        st.subheader("📑 Cluster Insights")
        st.markdown("""Patients with similar diagnosis keywords are grouped together.  
Structured features help separate acute vs. chronic/long-term management groups.""")

        for c in sorted(carematch.loc[mask, "cluster"].unique()):
            subset = carematch.loc[(carematch["cluster"] == c)]
            st.markdown(f"### 🔹 Cluster {int(c)} Summary")
            # Show top 5 diagnosis keywords
            top_diag = subset["diagnosis"].value_counts().head(5)
            st.dataframe(top_diag.reset_index().rename(columns={"index":"diagnosis","diagnosis":"count"}))
            # Numeric summaries
            st.write("**Avg Urgency:**", round(subset["urgency_score"].mean(), 2))
            st.write("**Avg Chronic Conditions:**", round(subset["chronic_conditions_count"].mean(), 2))
            st.write("**Mental Health Flag %:**", round(subset["mental_health_flag"].mean()*100, 2), "%")

        st.subheader("⏱️ Wait Time Distribution by Cluster")
        if "wait_time" in carematch.columns:
            fig12, ax12 = plt.subplots(figsize=(8,6))
            sns.boxplot(x="cluster", y="wait_time", data=carematch.loc[mask], ax=ax12)
            st.pyplot(fig12)

        st.subheader("🏥 Provider Specialty Distribution by Cluster")
        if "provider_specialty" in carematch.columns:
            fig13, ax13 = plt.subplots(figsize=(12,6))
            sns.countplot(x="cluster", hue="provider_specialty", data=carematch.loc[mask], ax=ax13)
            ax13.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            st.pyplot(fig13)

        st.subheader("📑 ***CLUSTER CONCLUSION***")
        st.markdown("Use these clusters as priors inside the Triage Assistant to guide specialty routing and expected wait times.")

    except Exception as e:
        # Show full traceback in the app while you iterate (remove later if you prefer)
        st.exception(e)
st.markdown("""***Key Takeaways***

- Clusters are not distinguished by wait time, but by provider specialty demand.

- Resource allocation should therefore focus on specialty coverage rather than purely reducing wait times.

- Cluster 1 and Cluster 3 represent the highest patient loads and may require more staffing and scheduling flexibility to balance demand.

- Clusters 0 and 2, though smaller, should not be overlooked as they might represent unique patient needs (e.g., targeted chronic conditions or specific demographics).""")

st.markdown("""***CONCLUSION***
- Our goal of the project is to improve wait time for patients’ appointment through analyzing the symptoms and the information about the patient such as zip code, provider specialty, age.
  However, our analysis shows no meaningful wait time improvement even with clustering, suggesting that more information needed for dataset over a long period of time, thus the robustness of the dataset would yield more meaningful insights during the data analysis process.""") 
# ======================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse import hstack

CLUSTER_COL = "cluster"          # you already create this
DIAG_COL = "diagnosis"           # from YAKE extraction

@st.cache_resource(show_spinner=True)
def build_cluster_assets(df: pd.DataFrame, k: int = 4):
    # 1) TF-IDF on diagnosis (short phrases -> very fast)
    mask = df[DIAG_COL].notnull()
    vectorizer = TfidfVectorizer(stop_words="english")
    X_text = vectorizer.fit_transform(df.loc[mask, DIAG_COL].astype(str))

    # 2) Structured -> scale
    scaler_struct = StandardScaler()
    X_struct = scaler_struct.fit_transform(
        df.loc[mask, ["urgency_score","chronic_conditions_count","mental_health_flag"]]
    )

    # 3) Fusion for clustering space
    X_cluster = hstack([X_text, X_struct])

    # 4) If you already labeled clusters, reuse them; else fit KMeans
    if CLUSTER_COL in df.columns and df[CLUSTER_COL].notnull().any():
        # respect existing labels but also fit a KMeans to get centroids for inference
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
        df.loc[mask, CLUSTER_COL] = kmeans.labels_

    # 5) Priors per cluster (specialty mix + avg wait + top diagnoses)
    priors = {}
    for c, sub in df.loc[mask].groupby(CLUSTER_COL):
        spec_share = sub["provider_specialty"].value_counts(normalize=True).to_dict()
        avg_wait = float(sub["wait_time"].mean()) if "wait_time" in sub else None
        top_diag  = sub[DIAG_COL].value_counts().head(5).index.tolist()
        priors[str(c)] = {"spec_share": spec_share, "avg_wait": avg_wait, "top_diag": top_diag}

    return vectorizer, scaler_struct, kmeans, priors

vectorizer, scaler_struct, kmeans, cluster_priors = build_cluster_assets(carematch, k=4)
import yake

kw_extractor = yake.KeywordExtractor(top=1, stopwords=None)

def infer_cluster_from_intake(summary: str, urgency: int, chronic: int, mh: int):
    # 1) Keyword from the new summary (same YAKE rule you used for the dataset)
    diag = None
    if summary and summary.strip():
        kws = kw_extractor.extract_keywords(summary)
        diag = kws[0][0] if kws else ""

    # 2) Vectorize with the SAME vectorizer/scaler used above
    Xt = vectorizer.transform([diag or ""])       # TF-IDF in diagnosis space
    Xs = scaler_struct.transform([[urgency, chronic, mh]])
    Xq = hstack([Xt, Xs])

    # 3) Predict cluster
    c = kmeans.predict(Xq)[0]
    return str(c), diag or ""
def search_similar_with_cluster(zip_code, urgency, chronic_count, mental_health, condition_summary, k=20):
    # build the FAISS query vector exactly like you already do
    results_global = search_similar(
        zip_code=str(zip_code).strip(),
        urgency=int(urgency),
        chronic_count=int(chronic_count),
        mental_health=int(mental_health),
        condition_summary=condition_summary,
        k=max(k, 50)  # fetch a bit more to allow sub-filtering
    )

    # infer cluster for this intake
    cluster_id, diag_kw = infer_cluster_from_intake(condition_summary, urgency, chronic_count, mental_health)

    # best slice: ZIP + cluster
    z = str(zip_code).strip()
    slice_zip_cluster = results_global[(carematch.loc[results_global.index, "zip_code"].astype(str).str.strip() == z) &
                                       (carematch.loc[results_global.index, CLUSTER_COL].astype(str) == cluster_id)]

    if len(slice_zip_cluster) >= max(5, k//2):
        use = slice_zip_cluster
    else:
        # fallback: same ZIP
        slice_zip_only = results_global[carematch.loc[results_global.index, "zip_code"].astype(str).str.strip() == z]
        use = slice_zip_only if len(slice_zip_only) >= max(5, k//2) else results_global

    # keep top-k by similarity
    use = use.sort_values("similarity", ascending=False).head(k).copy()
    use["__cluster_id__"] = cluster_id
    use["__diag_kw__"] = diag_kw
    return use, cluster_id
from collections import Counter

def summarize_with_priors(results: pd.DataFrame, cluster_id: str, alpha=0.7):
    # neighbor votes (similarity-weighted)
    spec_votes = Counter()
    if "provider_specialty" in results:
        for s, w in zip(results["provider_specialty"].astype(str), results["similarity"]):
            spec_votes[s] += float(w)

    # cluster prior
    prior = cluster_priors.get(cluster_id, {})
    prior_mix = prior.get("spec_share", {})
    prior_wait = prior.get("avg_wait", None)

    # blend
    combined = {}
    for s in set(list(spec_votes.keys()) + list(prior_mix.keys())):
        v = spec_votes.get(s, 0.0)
        p = prior_mix.get(s, 0.0)
        combined[s] = alpha * v + (1 - alpha) * p

    specialty = max(combined.items(), key=lambda kv: kv[1])[0] if combined else None
    provider  = results["assigned_provider_id"].mode().iloc[0] if "assigned_provider_id" in results and results["assigned_provider_id"].notna().any() else None
    wait      = float(results["wait_time"].mean()) if "wait_time" in results and results["wait_time"].notna().any() else prior_wait

    # short rationale
    diag_preview = ", ".join((cluster_priors.get(cluster_id, {}).get("top_diag") or [])[:3])
    rationale = f"Matches Cluster {cluster_id} (top dx: {diag_preview})" if diag_preview else f"Matches Cluster {cluster_id}"

    return provider, specialty, wait, rationale
if st.button("Generate Recommendation", use_container_width=True):
    if not condition_summary.strip():
        st.warning("⚠️ Please enter a condition summary."); st.stop()

    results, cluster_id = search_similar_with_cluster(
        zip_code=zip_code,
        urgency=int(urgency),
        chronic_count=int(chronic_count),
        mental_health=int(mental_health),
        condition_summary=condition_summary,
        k=k
    )

    provider, specialty, wait, rationale = summarize_with_priors(results, cluster_id, alpha=0.7)

    rec_text = generate_text_recommendation(
        inputs={
            "zip_code": zip_code,
            "urgency": urgency,
            "chronic_conditions_count": chronic_count,
            "mental_health_flag": mental_health,
            "summary": condition_summary[:200] + ("..." if len(condition_summary) > 200 else "")
        },
        provider=provider,
        specialty=specialty,
        wait=wait,
        results=results
    ) + (f"  \n_Reasoning_: {rationale}." if rationale else "")

    st.subheader("🔎 Recommendation")
    st.markdown(rec_text)

    # Quick facts + evidence (unchanged)
    ...


