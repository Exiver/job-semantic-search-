# 🚀 Semantic Job Search

A semantic job search engine built with sentence embeddings that matches your skills and experience to relevant job listings — going beyond keyword matching to understand the *meaning* of your query.

## Demo

Enter a natural language query like:
> *"Remote Python developer with AI and machine learning experience"*

And get ranked job results based on semantic similarity.

## How It Works

1. **Data** — Loads 500 job descriptions from the [`jacob-hugging-face/job-descriptions`](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions) dataset on HuggingFace
2. **Embeddings** — Encodes each job's title + description using `all-MiniLM-L6-v2` (a fast, lightweight sentence transformer)
3. **Search** — Encodes your query the same way, then ranks jobs by cosine similarity
4. **UI** — Displays top matches in a clean Streamlit interface

## Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Semantic embeddings (`all-MiniLM-L6-v2`) |
| `datasets` (HuggingFace) | Job description dataset |
| `numpy` | Embedding storage and math |
| `pandas` | Data manipulation |
| `streamlit` | Web UI |
| `torch` | Tensor operations for similarity scoring |

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/semantic-job-search.git
cd semantic-job-search
```

### 2. Create and activate a virtual environment

```bash
conda create -n ai-engineering python=3.11
conda activate ai-engineering
```

### 3. Install dependencies

```bash
pip install datasets sentence-transformers streamlit pandas numpy torch
```

### 4. Generate embeddings (run once)

Open and run `github_project.ipynb` in Jupyter. This will:
- Download 500 job descriptions
- Generate embeddings and save them to `job_enbeddings.npy`
- Save job data to `jobs.csv`

### 5. Launch the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
semantic-job-search/
│
├── github_project.ipynb   # Data prep & embedding generation
├── app.py                 # Streamlit web app
├── jobs.csv               # Saved job listings (generated)
├── job_enbeddings.npy     # Saved embeddings (generated)
└── README.md
```

## Example Queries

- `"Python developer machine learning"`
- `"Remote data scientist with NLP experience"`
- `"Frontend engineer React TypeScript"`
- `"Project manager agile scrum"`

## Notes

- The embeddings file (`job_enbeddings.npy`) is generated locally and not included in the repo — run the notebook first
- The model (`all-MiniLM-L6-v2`) will be downloaded automatically on first run (~80MB)
- For higher HuggingFace API rate limits, set a `HF_TOKEN` environment variable

## License

MIT
