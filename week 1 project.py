import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load saved data
df = pd.read_csv('jobs.csv')
embeddings = np.load('job_enbeddings.npy')  # matches your saved filename

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🚀 Semantic Job Search 2026")

query = st.text_input('What are you looking for?', 'Remote python developer with AI experience')

if query:
    query_vec = model.encode(query)
    scores = util.cos_sim(query_vec, embeddings)[0]
    df['score'] = scores.tolist()

    results = df.sort_values(by='score', ascending=False).head(5)

    for _, row in results.iterrows():
        with st.expander(f"{row['position_title']} (Match: {row['score']:.2%})"):
            st.write(f"**Company:** {row['company_name']}")
            st.write(row['job_description'][:500] + "...")