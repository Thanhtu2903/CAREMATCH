# -*- coding: utf-8 -*-
"""
Carematch Dashboard
Clean Streamlit version
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
carematch = pd.read_csv("carematch_requests.csv")

# --- Dashboard Title ---
st.title("📊 Carematch Dashboard")

# ========================
# SECTION 1: DATA OVERVIEW
# ========================
st.header("1️⃣ Data Overview")
st.write("Here is a preview of the dataset:")
st.write(carematch.head())

