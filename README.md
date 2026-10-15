# Healthcare-AI-Agent

A Streamlit app that answers health questions using a Retrieval-Augmented Generation (RAG) pipeline built on Google Gemini, Qdrant, and the [Agno](https://pypi.org/project/agno/) agent framework.

It also contains a standalone SVM heart-disease classification script.

## What it does

`Report_Generator.py` is the whole app (single file, Streamlit script — no package, no CLI). Three tabs:

**Chat** — the RAG pipeline, run per question:
1. A *Query Rewriter* agent expands shorthand into medical terms (e.g. `BP` → `Blood Pressure`).
2. Retrieval from a Qdrant vector store using a similarity-score threshold (default `0.7`, adjustable in the sidebar).
3. If nothing relevant is found (or you force it with the 🌐 toggle) and web search is enabled, an Exa-backed *Web Search Agent* queries a domain allowlist — defaults to `mayoclinic.org`, `nih.gov`, `cdc.gov`, `webmd.com`, `medlineplus.gov`.
4. A *Healthcare Assistant* agent writes the final answer, prefixed with the retrieved context.

Documents get into the vector store two ways: upload a PDF, or paste a URL. Both are loaded with LangChain loaders, split with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, and embedded with Gemini `text-embedding-004`.

**Health Records** — a form for name, age, gender, height, weight, blood pressure, heart rate, and blood sugar. Note: the "Save Health Data" button only shows a success message; nothing is persisted.

**Medicine Schedule** — a text area for `Medicine Name, HH:MM` lines. The parsed schedule is validated and confirmed, but no reminder is actually scheduled or emailed; the app only warns when the Gmail credentials are missing.

`ml prediction.py` is separate and unrelated to the app: it downloads the UCI Heart Disease dataset, trains an RBF-kernel SVM, prints accuracy, and plots a confusion matrix. It is a Colab export and does not save a model.

## Requirements

- Python (unverified — no version is pinned anywhere in the repo)
- `global_requirements.txt` — a full `pip freeze`, 261 packages, UTF-16LE encoded. Key entries:

  | Package | Version |
  | --- | --- |
  | streamlit | 1.42.0 |
  | agno | 1.1.1 |
  | langchain | 0.3.19 |
  | langchain-community | 0.3.17 |
  | langchain-qdrant | 0.2.0 |
  | qdrant-client | 1.13.2 |
  | google-generativeai | 0.8.4 |
  | exa_py | 1.8.9 |
  | scikit-learn | 1.6.1 |
  | pandas | 2.2.3 |
  | joblib | 1.4.2 |
  | beautifulsoup4 | 4.12.3 |

Because the file is UTF-16 encoded, install it with a converter first:

```bash
iconv -f UTF-16LE -t UTF-8 global_requirements.txt > requirements.txt
pip install -r requirements.txt
```

Some pinned transitive versions (notably `pydantic==1.10.21`) may conflict with each other on a modern interpreter; the app was last touched in May 2025 and has not been re-pinned since.

## Configuration

All settings are read from `.env` via `python-dotenv` at import time, and every key can also be typed into the app's sidebar at runtime (the sidebar value wins).

| Variable | Required for | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Everything — the app shows a warning and stops without it | Also used for embeddings |
| `QDRANT_API_KEY` | Document upload and retrieval | |
| `QDRANT_URL` | Document upload and retrieval | e.g. `https://your-cluster.cloud.qdrant.io:6333` |
| `EXA_API_KEY` | Web search fallback | Only read when the sidebar checkbox is on |
| `GMAIL_USER` | Unused in practice | |
| `GMAIL_APP_PASSWORD` | Unused in practice | |
| `RECIPIENT_EMAIL` | Unused in practice | |

Vector store collection name: `gemini-thinking-agent-agno`.

> **Security:** `.env` is committed to this repository with real values. Treat every key in it as leaked — rotate them, then run `git rm --cached .env` and add `.env` to `.gitignore` (it currently contains only `.qodo`).

## Running

```bash
streamlit run Report_Generator.py
```

`GOOGLE_API_KEY` is the only hard requirement to get past the initial warning; everything else degrades gracefully (no Qdrant → no document search, no Exa → no web fallback).

The SVM script runs standalone:

```bash
python "ml prediction.py"
```

## Known gaps

- `load_and_predict()` in `Report_Generator.py` expects `svm_model.pkl` and `scaler.pkl` in the working directory. Neither file is in the repo, and `ml prediction.py` never saves them. The function is currently unreferenced by the Streamlit UI.
- The Health Records and Medicine Schedule tabs are UI scaffolding, not implemented features.
- `requests`, `smtplib`, and `Tuple` are imported but unused.
- No tests, no CI.

## License

None specified.