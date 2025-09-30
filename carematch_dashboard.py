
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

# === Load Dataset ===
carematch = pd.read_csv("carematch_requests.csv")

# === Dashboard Title ===
st.title("📊 Carematch Dashboard")
# === Introduction / Project Background ===
st.header("🏥 Project Background")
st.markdown("""
**CareMatch Health** is a regional healthcare network serving a diverse patient population across both urban and suburban communities.  
Patients submit appointment requests and complete intake forms through the organization’s digital platforms.  

Although CareMatch holds a large volume of patient and operational data, it has not yet implemented advanced analytics or AI-powered tools to derive value from this information.  
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

with col2:
    st.subheader("Chronic Conditions Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(carematch['chronic_conditions_count'], bins=20, kde=False, color='blue', ax=ax2)
    st.pyplot(fig2)

# === Boxplots ===
st.header("📊 Wait Time by Categories")

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
for c in range(k):
    subset = carematch[carematch["cluster"] == c]
    st.markdown(f"### 🔹 Cluster {c} Summary")
    st.write(subset["diagnosis"].value_counts().head(5))
    st.write("**Avg Urgency:**", round(subset["urgency_score"].mean(), 2))
    st.write("**Avg Chronic Conditions:**", round(subset["chronic_conditions_count"].mean(), 2))
    st.write("**Mental Health Flag %:**", round(subset["mental_health_flag"].mean()*100, 2), "%")

st.subheader("⏱️ Wait Time Distribution by Cluster")
fig12, ax12 = plt.subplots(figsize=(8,6))
sns.boxplot(x="cluster", y="wait_time", data=carematch, ax=ax12)
st.pyplot(fig12)

st.subheader("🏥 Provider Specialty Distribution by Cluster")
fig13, ax13 = plt.subplots(figsize=(12,6))
sns.countplot(x="cluster", hue="provider_specialty", data=carematch, ax=ax13)
st.pyplot(fig13)
