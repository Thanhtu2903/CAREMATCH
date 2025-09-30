
# -*- coding: utf-8 -*-
"""
Carematch Dashboard
Clean Streamlit version
"""

# === Imports ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.sparse import hstack
import numpy as np, pandas as pd, streamlit as st, faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from collections import Counter


# === Load Dataset ===
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
st.markdown("""***Data Preprocessing ***: 
The dataset contained free-text entries under the column “condition_summary.”
Since raw text entries often contained verbose sentences, we first applied keyword extraction. 
We use YAKE package (Yet Another Keyword Extractor) was used to extract the most important terms from each condition summary. 
These extracted keywords were used as the primary diagnosis terms. 
Example: Raw Condition Summary: “Ongoing depression and emotional instability, need therapy.” Extracted Diagnosis: “ongoing depression.” 
This ensured a consistent representation of patient conditions for clustering.""")
st.header("🩺 Keyword Extraction from Condition Summaries")
kw_extractor = yake.KeywordExtractor(top=1, stopwords=None)
def extract_keyword(text):
    if pd.isnull(text):
        return None
    keywords = kw_extractor.extract_keywords(text)
    return keywords[0][0] if keywords else None
carematch["diagnosis"] = carematch["condition_summary"].apply(extract_keyword)
st.subheader("Sample Condition Summaries with Diagnosis Keyword")
st.dataframe(carematch[["condition_summary","diagnosis"]].head(50))
keyword_counts = carematch['diagnosis'].value_counts().reset_index()
keyword_counts.columns = ["diagnosis_keyword","count"]
st.subheader("Most Frequent Diagnosis Keywords")
st.dataframe(keyword_counts.head(20))
fig9, ax9 = plt.subplots(figsize=(10,6))
sns.barplot(data=keyword_counts.head(15), x="count", y="diagnosis_keyword", palette="viridis", ax=ax9)
st.pyplot(fig9)

# === Clustering ===
st.header("🤖 Patient Clustering Analysis")
st.markdown("""***Clustering method:***
Beyond diagnosis extraction, we created additional features to better capture patient needs: urgency_score, chronic_conditions_count, and mental_health_flag. Together with the extracted “diagnosis”, these features formed the input for clustering analysis. To group patients into meaningful cohorts, we applied k-means clustering using scikit-learn package. The steps include the following:  

- ***Vectorization:*** The “diagnosis” text was converted into numeric vectors using TF-IDF (term frequency–inverse document frequency).  

- ***Feature Combination:*** “urgency_score”, “chronic_conditions_count”, “mental_health_flag”) were concatenated with text embeddings.  

- ***Optimal Cluster Selection:*** The elbow method was used to determine the appropriate number of clusters by plotting the within- sum-of-squares (WSS) across different values of. We found that k=4 is the optimal number. 

Cluster Labeling: Each patient was assigned to a cluster, which was then merged back with the original dataset for further analysis. """)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(carematch["diagnosis"].dropna())
structured = carematch.loc[carematch["diagnosis"].notnull(), ["urgency_score","chronic_conditions_count","mental_health_flag"]]
scaler = StandardScaler()
X_structured = scaler.fit_transform(structured)
X_cluster = hstack([X, X_structured])

st.header("📉 Elbow Method for Optimal k")
inertia = []
K = range(2,11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)
fig10, ax10 = plt.subplots(figsize=(8,6))
ax10.plot(K, inertia, "bo-")
ax10.set_xlabel("Number of clusters (k)")
ax10.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
st.pyplot(fig10)

st.sidebar.subheader("⚙️ Clustering Parameters")
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
carematch.loc[carematch["diagnosis"].notnull(),"cluster"] = kmeans.fit_predict(X_cluster)

st.subheader("📊 PCA Visualization of Clusters")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())
fig11, ax11 = plt.subplots(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=carematch.loc[carematch["diagnosis"].notnull(),"cluster"], palette="tab10", ax=ax11)
st.pyplot(fig11)

st.subheader("📑 Cluster Insights")
st.markdown("""*** Insignt:***
Patients with similar diagnoses (e.g., depression, anxiety) were grouped together. High-urgency cases formed distinct clusters (e.g., ongoing depression, sudden vision blur). 
Chronic condition count played a role in differentiating patient profiles, especially in separating acute vs. long-term management needs.""")
for c in range(k):
    subset = carematch[carematch["cluster"] == c]
    st.markdown(f"### 🔹 Cluster {c} Summary")
    st.write(subset["diagnosis"].value_counts().head(5))
    st.write("**Avg Urgency:**", round(subset["urgency_score"].mean(), 2))
    st.write("**Avg Chronic Conditions:**", round(subset["chronic_conditions_count"].mean(), 2))
    st.write("**Mental Health Flag %:**", round(subset["mental_health_flag"].mean()*100, 2), "%")
    
st.subheader("📑***CLUSTER CONCLUSION***")

st.subheader("⏱️ Wait Time Distribution by Cluster")
fig12, ax12 = plt.subplots(figsize=(8,6))
sns.boxplot(x="cluster", y="wait_time", data=carematch, ax=ax12)
st.pyplot(fig12)
st.markdown("""***Cluster 3*** has shorter wait time than other clusters. 
This could be explained by the diagnosis symptoms such as feeling dizzy, high blood pressure, sports injury, and experiencing shortness of breath that all are externally visible, thus clinics tend to prioritize and speed up these cases.""")
st.subheader("🏥 Provider Specialty Distribution by Cluster")
fig13, ax13 = plt.subplots(figsize=(12,6))
sns.countplot(x="cluster", hue="provider_specialty", data=carematch, ax=ax13)
st.pyplot(fig13)
st.markdown("""***Key Takeaways***

- Clusters are not distinguished by wait time, but by provider specialty demand.

- Resource allocation should therefore focus on specialty coverage rather than purely reducing wait times.

- Cluster 1 and Cluster 3 represent the highest patient loads and may require more staffing and scheduling flexibility to balance demand.

- Clusters 0 and 2, though smaller, should not be overlooked as they might represent unique patient needs (e.g., targeted chronic conditions or specific demographics).""")

st.markdown("""***CONCLUSION***
- Our goal of the project is to improve wait time for patients’ appointment through analyzing the symptoms and the information about the patient such as zip code, provider specialty, age.
  However, our analysis shows no meaningful wait time improvement even with clustering, suggesting that more information needed for dataset over a long period of time, thus the robustness of the dataset would yield more meaningful insights during the data analysis process.""") 
# ======================
# Settings
# --------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
STRUCT_COLS = ["urgency_score", "chronic_conditions_count", "mental_health_flag"]  # must be 3 and in this order
TEXT_COL = "condition_summary"
ZIP_COL = "zip_code"
PROVIDER_COL = "assigned_provider_id"
SPEC_COL = "provider_specialty"
WAIT_COL = "wait_time"

# Optional: map your actual column names to the required ones
RENAME_MAP = {
    # "urgency": "urgency_score",
    # "chronic_count": "chronic_conditions_count",
    # "mental_health": "mental_health_flag",
}
# Apply rename (only keys that exist)
if "carematch" in globals():
    carematch = carematch.rename(columns={k: v for k, v in RENAME_MAP.items() if k in carematch.columns})

# -------------------
# Basic data hygiene
# -------------------
def _ensure_columns(df: pd.DataFrame):
    required = [ZIP_COL, TEXT_COL, SPEC_COL] + STRUCT_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in carematch: {missing}")
    # coerce numerics
    df[STRUCT_COLS] = (df[STRUCT_COLS]
                       .apply(pd.to_numeric, errors="coerce")
                       .fillna(0.0)
                       .astype("float32"))
    if WAIT_COL in df.columns:
        df[WAIT_COL] = pd.to_numeric(df[WAIT_COL], errors="coerce").astype("float32")
    # condition_summary to str
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    # zip as string (normalize type for indexing)
    df[ZIP_COL] = df[ZIP_COL].astype(str).str.strip()
    return df

if "carematch" in globals():
    carematch = _ensure_columns(carematch)

# -----------------------
# Build all app resources
# -----------------------
@st.cache_resource(show_spinner=True)
def build_assets(df: pd.DataFrame):
    # 1) text embeddings
    model = SentenceTransformer(EMBED_MODEL_NAME)
    X_text = model.encode(df[TEXT_COL].tolist(), show_progress_bar=False,
                          convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(X_text)

    # 2) structured features (3 features, fixed order)
    scaler = StandardScaler().fit(df[STRUCT_COLS].astype("float32"))
    X_struct = scaler.transform(df[STRUCT_COLS].astype("float32")).astype("float32")

    # 3) combined matrix
    X_all = np.hstack([X_text, X_struct]).astype("float32")
    faiss.normalize_L2(X_all)

    # 4) global FAISS index
    d = X_all.shape[1]
    global_index = faiss.IndexFlatIP(d)
    global_index.add(X_all)

    # 5) per-zip FAISS indices (pre-filter via zip)
    per_zip_index = {}
    per_zip_rows = {}
    for z, sub in df.groupby(ZIP_COL, sort=False):
        rows = sub.index.to_numpy()
        per_zip_rows[z] = rows
        X_sub = X_all[rows]
        idx = faiss.IndexFlatIP(d)
        idx.add(X_sub)
        per_zip_index[z] = idx

    return model, scaler, global_index, per_zip_index, per_zip_rows

if "carematch" in globals():
    model, scaler, global_index, per_zip_index, per_zip_rows = build_assets(carematch)

# -------------------------
# Core retrieval function
# -------------------------
def search_similar(zip_code: str, urgency: int, chronic_count: int, mental_health: int,
                   condition_summary: str, k: int = 20):
    # 1) build query vector (same 3 features in same order)
    q_struct = np.array([[urgency, chronic_count, mental_health]], dtype="float32")
    q_struct = scaler.transform(q_struct).astype("float32")

    q_text = model.encode([condition_summary], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_text)

    q = np.hstack([q_text, q_struct]).reshape(1, -1)

    # 2) choose index: per-zip first, else global
    z = str(zip_code).strip()
    if z in per_zip_index and len(per_zip_rows[z]) >= max(5, k//2):
        D, I = per_zip_index[z].search(q, k)
        # map local indices back to original DataFrame rows
        row_ids = per_zip_rows[z][I[0]]
    else:
        D, I = global_index.search(q, k)
        row_ids = I[0]

    results = carematch.iloc[row_ids].copy()
    results["similarity"] = D[0]
    return results

# -----------------------------------
# Summarize into a "recommendation"
# -----------------------------------
def summarize_recommendation(results: pd.DataFrame):
    spec = None
    provider = None
    wait = None

    if not results.empty:
        # specialty by frequency then by average similarity
        if SPEC_COL in results.columns:
            spec_counts = Counter(results[SPEC_COL].dropna().astype(str))
            if spec_counts:
                # tie-breaker by mean similarity inside the specialty
                top_spec, _ = max(
                    spec_counts.items(),
                    key=lambda kv: (kv[1], results.loc[results[SPEC_COL] == kv[0], "similarity"].mean())
                )
                spec = top_spec

        if PROVIDER_COL in results.columns and results[PROVIDER_COL].notna().any():
            provider = results[PROVIDER_COL].mode().iloc[0]

        if WAIT_COL in results.columns and results[WAIT_COL].notna().any():
            wait = float(results[WAIT_COL].mean())

    return provider, spec, wait

# -----------------------------------
# Optional: LLM generation (OpenAI)
# -----------------------------------
def generate_text_recommendation(inputs, provider, specialty, wait, results):
    """
    If OPENAI_API_KEY is set, use OpenAI for a nicer summary.
    Otherwise, return a rule-based paragraph.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            few_examples = (
                "You are triaging patient intake into provider specialties. "
                "You must be concise, non-clinical, and avoid diagnoses. "
                "Use urgency, chronic count, mental health flag, and similar-case evidence.\n"
            )
            top_rows = results.head(5)[[SPEC_COL, WAIT_COL, "similarity"]].fillna("").to_dict("records")
            prompt = (
                f"{few_examples}"
                f"Inputs: {inputs}. Suggested specialty: {specialty}. "
                f"Estimated average wait: {None if wait is None else round(wait,1)} days. "
                f"Top similar cases (specialty, wait, similarity): {top_rows}.\n"
                "Write 3–5 sentences recommending which specialty to book with and why, "
                "mentioning urgency and providing a next step."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=180,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass  # fall back to rule-based

    # Fallback: rule-based concise text
    lines = []
    if specialty:
        lines.append(f"Recommended specialty: **{specialty}** based on the most similar past cases.")
    else:
        lines.append("No clear specialty majority from similar cases; suggest routing to **General Practice / Intake Nurse** for first review.")
    if wait is not None:
        lines.append(f"Estimated wait time from similar cases: **{round(wait,1)} days**.")
    lines.append("Next step: create an appointment request with the suggested specialty, and add notes from the condition summary for prioritization.")
    return " ".join(lines)

# ======================
# Streamlit UI
# ======================
st.title("🤖 Internal AI Triage Assistant")

if "carematch" not in globals():
    st.error("Dataset `carematch` is not loaded. Please load it before running the app.")
    st.stop()

st.caption("Searches similar historical cases (text + structured signals) and proposes a specialty/provider.")

st.subheader("📝 Patient Intake Form")
col1, col2 = st.columns(2)
with col1:
    zip_code = st.text_input("📍 Zip Code", value="")
    urgency = st.selectbox("⚡ Urgency Score", sorted(carematch["urgency_score"].dropna().unique().tolist()))
with col2:
    chronic_count = st.number_input("🩺 Chronic Conditions Count", min_value=0, max_value=20, step=1, value=0)
    mental_health = st.selectbox("🧠 Mental Health Flag", [0, 1])

condition_summary = st.text_area("💬 Patient Condition Summary", height=140, placeholder="Brief symptoms, duration, known conditions...")

k = st.slider("Top-K similar cases to consider", 5, 100, 20, 5)

if st.button("Generate Recommendation", use_container_width=True):
    if not condition_summary.strip():
        st.warning("⚠️ Please enter a condition summary.")
        st.stop()

    # Retrieval
    results = search_similar(
        zip_code=str(zip_code).strip() if zip_code else "",
        urgency=int(urgency),
        chronic_count=int(chronic_count),
        mental_health=int(mental_health),
        condition_summary=condition_summary,
        k=k
    )

    # Summaries
    provider, specialty, wait = summarize_recommendation(results)

    # NL recommendation
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
    )

    st.subheader("🔎 Recommendation")
    st.markdown(rec_text)

    # Quick facts
    left, right = st.columns(2)
    with left:
        st.markdown(f"- **Suggested Provider ID:** `{provider}`" if provider else "- **Suggested Provider ID:** _no clear majority_")
        st.markdown(f"- **Specialty:** `{specialty}`" if specialty else "- **Specialty:** _no clear majority_")
        st.markdown(f"- **Avg Wait (days):** `{round(wait,1)}`" if wait is not None else "- **Avg Wait:** _n/a_")
    with right:
        st.markdown(f"- **Urgency:** {urgency}")
        st.markdown(f"- **Chronic Count:** {chronic_count}")
        st.markdown(f"- **Mental Health:** {mental_health}")

    # Similar cases table
    st.subheader("📋 Similar Past Cases")
    show_cols = [c for c in [TEXT_COL, PROVIDER_COL, SPEC_COL, WAIT_COL, "similarity"] if c in results.columns]
    st.dataframe(results[show_cols].reset_index(drop=True))

    # Specialty distribution
    if SPEC_COL in results.columns and not results.empty:
        spec_counts = results[SPEC_COL].value_counts().reset_index()
        spec_counts.columns = ["specialty", "count"]
        st.bar_chart(spec_counts.set_index("specialty"))


