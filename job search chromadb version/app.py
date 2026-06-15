import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

st.set_page_config(page_title="Agentic job Search", page_icon="🚀", layout="wide")

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path="./job_chroma_db")
    sentence_transformer_ef =embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = "all-MiniLM-L6-v2",
    )
    collection = client.get_or_create_collection(
        name = "agentic_job_search",
        embedding_function = sentence_transformer_ef
    )
    return collection

@st.cache_resource
def load_llm():
    llm = ChatOllama(model="llama3.1:8b", tempreature=0.3)
    prompt = ChatPromptTemplate.from_template(
        template= """  
        the user is looking for a job with these characteristics :"{query}"
        you found this job :"{job_title}"
        Description snippet: {job_desc}
        in exactly 2 short, punchy sentences, explain to the user WHY this specific job is a good match for what they are looking for. Focus on the skills and intent    """
    )
    return llm, prompt

collection = load_db()
llm, prompt = load_llm()

st.title("Agentic job search engine (local)")
st.markdown("search via vector database (chromadb) + local llama3 ")

with st.sidebar:
    st.header("search parameter")
    top_k = st.slider("results to fetch ", min_value=1, max_value=10, value=3)
    fillter_location = st.text_input("filter by location ", placeholder="e.g, cairo, new york")

query = st.text_input("describe your ideal rule, skills, and goals:", "building RAG pipeline, deploying local llms, and python engineering")

if query:
    with st.spinner("Searching vector space..."):
        where_clause=None
        if fillter_location:
            where_clause= {"location": {"$contains": fillter_location}}

        results =collection.query(
            query_texts= [query],
            n_results= top_k,
            where= where_clause
        )

    st.success(f"retrieved top{top_k} matches ")


    for i in range(top_k):
        meta = results['metadatas'][0][i]
        doc =results['documents'][0][i]
        job_title = meta['position_title']
        company = meta['company_name']


        formatted_prompt = prompt.format(query=query, job_title=job_title, job_desc=doc[:1000])
        rationale = llm.invoke(formatted_prompt).content


        with st.container():
            st.subheader(f"{job_title} at {company}")
            st.info(f"why its a match {rationale}")

            with st.expander("view full job describtion"):
                st.write(doc)
            st.markdown("---")
