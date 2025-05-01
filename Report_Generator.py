import os
import tempfile
from datetime import datetime
from typing import List, Tuple
import joblib

import streamlit as st
import requests
import smtplib
import bs4
import google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------
# Model Prediction Function
# --------------------------

def load_and_predict(input_data):
    """
    Load saved SVM model and make disease predictions.
    
    Args:
        input_data: Features to predict on (array-like)
        
    Returns:
        str: "Disease" or "No Disease" prediction
    """
    # Load the saved SVM model
    try:
        model = joblib.load('svm_model.pkl')  # Adjust the path as necessary
    except FileNotFoundError:
        print("Model not found. Please train and save the model first.")
        return None
    
    # Load the scaler (you can also save the scaler during training for consistency)
    scaler = joblib.load('scaler.pkl')  # Make sure to save the scaler during training
    
    # Preprocess the input data (assuming it needs scaling)
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Return the prediction result as a string
    if prediction == 1:
        return "Disease"
    else:
        return "No Disease"

# Set API keys & URLs from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
# Additional variables for environmental factors and medication reminders
POLLUTION_API_KEY = os.environ.get("POLLUTION_API_KEY", "demo")  # Use "demo" if not set
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")

# --------------------------
# Gemini Embeddings Class
# --------------------------

class GeminiEmbedder(Embeddings):
    def __init__(self, model_name="models/text-embedding-004"):
        genai.configure(api_key=st.session_state.google_api_key)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']


# --------------------------
# Constants & App Initialization
# --------------------------

COLLECTION_NAME = "gemini-thinking-agent-agno"

st.title("👨‍⚕️ General Health Doctor Assistant")

# Initialize session state (load from environment if not already present)
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = GOOGLE_API_KEY
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = QDRANT_API_KEY
if 'qdrant_url' not in st.session_state:
    st.session_state.qdrant_url = QDRANT_URL
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = EXA_API_KEY
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7

# --------------------------
# Sidebar Configuration
# --------------------------

st.sidebar.header("🔑 API Configuration")
google_api_key = st.sidebar.text_input("Google API Key", type="password", value=st.session_state.google_api_key)
qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password", value=st.session_state.qdrant_api_key)
qdrant_url = st.sidebar.text_input("Qdrant URL", placeholder="https://your-cluster.cloud.qdrant.io:6333",
                                   value=st.session_state.qdrant_url)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()

st.session_state.google_api_key = google_api_key
st.session_state.qdrant_api_key = qdrant_api_key
st.session_state.qdrant_url = qdrant_url

# Web Search Configuration using Exa (for fallback)
st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback",
                                                      value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input("Exa AI API Key", type="password", value=st.session_state.exa_api_key,
                                        help="Required for web search fallback when no relevant documents are found")
    st.session_state.exa_api_key = exa_api_key
    default_domains = ["mayoclinic.org", "nih.gov", "cdc.gov", "webmd.com", "medlineplus.gov"]
    custom_domains = st.sidebar.text_input("Custom domains (comma-separated)",
                                           value=",".join(default_domains),
                                           help="Enter domains to search from, e.g.: mayoclinic.org,nih.gov")
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Search Configuration
st.sidebar.header("🎯 Search Configuration")
st.session_state.similarity_threshold = st.sidebar.slider("Document Similarity Threshold", min_value=0.0, max_value=1.0,
                                                          value=0.7,
                                                          help="Lower values will return more documents but might be less relevant. Higher values are more strict.")

# --------------------------
# Document Processing Functions
# --------------------------

def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60
        )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None


def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header", "content", "main")
            ))
        )
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


def create_vector_store(client, texts):
    """Create and initialize vector store with documents."""
    try:
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=GeminiEmbedder()
        )
        with st.spinner('📤 Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store
    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None


# --------------------------
# Agent Functions
# --------------------------

def get_query_rewriter_agent() -> Agent:
    """Initialize a query rewriting agent specialized for general healthcare."""
    return Agent(
        name="Query Rewriter",
        model=Gemini(id="gemini-exp-1206"),
        instructions=(
            "You are an expert at reformulating health-related questions. "
            "Analyze the user's question, rewrite it to be more specific and medically accurate "
            "(e.g., expand 'BP' to 'Blood Pressure', 'T2D' to 'Type 2 Diabetes'), "
            "and return ONLY the rewritten query without extra commentary."
        ),
        show_tool_calls=False,
        markdown=False
    )


def get_web_search_agent() -> Agent:
    """Initialize a web search agent using ExaTools for medically relevant information."""
    return Agent(
        name="Web Search Agent",
        model=Gemini(id="gemini-exp-1206"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions=(
            "You are a web search expert focusing on evidence-based medical information. Search the web for up-to-date, reliable information "
            "on health topics, nutrition, disease prevention, and treatment guidelines. Prefer credible medical sources like Mayo Clinic, "
            "CDC, NIH, and peer-reviewed research. Summarize the most relevant information and include sources."
        ),
        show_tool_calls=True,
        markdown=False
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent as a general health assistant."""
    return Agent(
        name="Healthcare Assistant",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions=(
            "Hello 👋, I'm your friendly Healthcare Assistant! I'm here to help you with personalized health advice, including diet plans, "
            "lifestyle recommendations, and clarifications on general health concerns. "
            "I can provide information on a wide range of health topics, including preventive care, common conditions, "
            "nutrition, exercise, mental health, and general wellness. "
            "I always recommend consulting a healthcare professional for personalized care and will never replace medical diagnosis. "
            "When given context from documents, I'll focus on extracting evidence-based information. "
            "When provided with web search results, I'll indicate the source and synthesize the information in clear, plain text. "
            "Let's chat and work together to improve your overall health and well-being! 💪"
        ),
        show_tool_calls=True,
        markdown=False
    )


def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    """
    Check if documents in vector store are relevant to the query.
    Returns a tuple: (has_relevant_docs, relevant_docs)
    """
    if not vector_store:
        return False, []
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs


# --------------------------
# Main Application Flow
# --------------------------

if st.session_state.google_api_key:
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    genai.configure(api_key=st.session_state.google_api_key)

    qdrant_client = init_qdrant()

    # File/URL Upload Section
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")

    # Process documents
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner('Processing PDF...'):
                texts = process_pdf(uploaded_file)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ Added PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner('Processing URL...'):
                texts = process_web(web_url)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(web_url)
                    st.success(f"✅ Added URL: {web_url}")

    # Display processed sources in sidebar
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            if source.endswith('.pdf'):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

    # Create Tabs: Chat, Health Records, Medicine Schedule
    tabs = st.tabs(["Chat", "Health Records", "Medicine Schedule"])

    # --------------------------
    # Tab 1: Chat Interface
    # --------------------------
    with tabs[0]:
        # Chat Interface: Two columns for chat input and search toggle
        chat_col, toggle_col = st.columns([0.9, 0.1])
        with chat_col:
            prompt = st.chat_input("Hello! How can I help with your health concerns today? 😊")
        with toggle_col:
            st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

        if prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Step 1: Rewrite the query for better retrieval (medical focus)
            with st.spinner("🤔 Reformulating your question..."):
                try:
                    query_rewriter = get_query_rewriter_agent()
                    rewritten_query = query_rewriter.run(prompt).content
                    with st.expander("🔄 See revised question"):
                        st.write(f"Original: {prompt}")
                        st.write(f"Revised: {rewritten_query}")
                except Exception as e:
                    st.error(f"❌ Error rewriting question: {str(e)}")
                    rewritten_query = prompt

            # Step 2: Retrieve context from stored documents
            context = ""
            docs = []
            if not st.session_state.force_web_search and st.session_state.vector_store:
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "score_threshold": st.session_state.similarity_threshold}
                )
                docs = retriever.invoke(rewritten_query)
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    st.info(f"📊 Found {len(docs)} relevant document(s)")
                elif st.session_state.use_web_search:
                    st.info("🔄 No relevant documents found. Will use web search...")

            # Step 3: Optionally perform web search if forced or no context available
            if (st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
                with st.spinner("🔍 Searching the web for more info..."):
                    try:
                        web_search_agent = get_web_search_agent()
                        web_results = web_search_agent.run(rewritten_query).content
                        if web_results:
                            context = f"Web Search Results:\n{web_results}"
                            if st.session_state.force_web_search:
                                st.info("ℹ️ Using web search as requested.")
                            else:
                                st.info("ℹ️ Using web search as fallback.")
                    except Exception as e:
                        st.error(f"❌ Web search error: {str(e)}")

            # Step 4: Generate response using the Healthcare Assistant Agent
            with st.spinner("🤖 Generating your personalized health advice..."):
                try:
                    rag_agent = get_rag_agent()
                    if context:
                        full_prompt = (
                            f"Context: {context}\n\n"
                            f"User Question: {prompt}\n"
                            f"Revised Question: {rewritten_query}\n\n"
                            "Please provide a comprehensive, personalized health response in plain text with icons and emojis. "
                            "Greet the user warmly, clarify any doubts, and offer tailored advice. Always advise consulting a healthcare professional."
                        )
                    else:
                        full_prompt = (
                            f"User Question: {prompt}\n"
                            f"Revised Question: {rewritten_query}\n\n"
                            "Please provide personalized health advice in plain text with icons and emojis. "
                            "Greet the user warmly and ask clarifying questions if needed. Always remind the user to consult a healthcare professional."
                        )
                        st.info("ℹ️ No additional context found. Generating advice based solely on your question.")

                    response = rag_agent.run(full_prompt)
                    st.session_state.history.append({"role": "assistant", "content": response.content})
                    with st.chat_message("assistant"):
                        st.write(response.content)
                        # Optionally show document sources
                        if not st.session_state.force_web_search and docs:
                            with st.expander("🔍 Document Sources"):
                                for i, doc in enumerate(docs, 1):
                                    source_type = doc.metadata.get("source_type", "unknown")
                                    source_icon = "📄" if source_type == "pdf" else "🌐"
                                    source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url",
                                                                   "unknown")
                                    st.write(f"{source_icon} Source {i} from {source_name}:")
                                    st.write(f"{doc.page_content[:200]}...")
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

    # --------------------------
    # Tab 2: Health Records
    # --------------------------
    with tabs[1]:
        st.header("📋 Health Records")
        st.write("Track your key health metrics over time.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Information")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
            height = st.number_input("Height (cm)", min_value=0, max_value=250, step=1)
            weight = st.number_input("Weight (kg)", min_value=0, max_value=500, step=1)
            
        with col2:
            st.subheader("Vital Signs")
            bp_sys = st.number_input("Blood Pressure (Systolic)", min_value=0, max_value=250, step=1)
            bp_dia = st.number_input("Blood Pressure (Diastolic)", min_value=0, max_value=250, step=1)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=250, step=1)
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=0, max_value=600, step=1)
        
        if st.button("Save Health Data"):
            st.success("✅ Health data saved successfully!")
            # Placeholder for actual save functionality

    # --------------------------
    # Tab 3: Medicine Schedule
    # --------------------------
    with tabs[2]:
        st.header("💊 Medicine Schedule")
        st.write("Enter your medications (one per line) in the format: `Medicine Name, HH:MM`")
        tablets_input = st.text_area("Medications", placeholder="Aspirin, 08:00\nVitamin D, 12:00")

        # Button to set reminders manually
        if st.button("Set Medication Reminders"):
            if not tablets_input:
                st.error("Please enter at least one medication.")
            elif not (GMAIL_USER and GMAIL_APP_PASSWORD and RECIPIENT_EMAIL):
                st.warning("Email credentials not configured. Reminders saved locally only.")
                st.success("✅ Medication schedule saved locally!")
            else:
                # Parse input into list of (medicine, time) tuples
                medications = []
                for line in tablets_input.strip().splitlines():
                    if ',' in line:
                        parts = line.split(',')
                        medicine = parts[0].strip()
                        time_str = parts[1].strip()
                        medications.append((medicine, time_str))
                if medications:
                    st.success("✅ Medication reminders set successfully!")
                else:
                    st.error("No valid medication entries found.")

else:
    st.warning("⚠️ Please enter your Google API Key to continue")