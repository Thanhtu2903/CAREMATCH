# -*- coding: utf-8 -*-
"""
Carematch Dashboard
Clean Streamlit version
"""

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
from pathlib import Path
import streamlit as st


st.markdown(""" ***GROUP 4***: TU PHAM & MINH NGUYEN""")
# === Dashboard Title ===
st.title("📊 Carematch Dashboard")

# === Load Dataset ===
carematch = pd.read_csv("carematch_requests.csv")
