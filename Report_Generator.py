import os
import tempfile
from datetime import datetime
from typing import List, Tuple

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

# Set API keys & URLs from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
# Additional variables for pollution and email reminder
POLLUTION_API_KEY = os.environ.get("POLLUTION_API_KEY", "demo")  # Use "demo" if not set
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")


# --------------------------
# Helper Functions for Live Location & Pollution
# --------------------------

def get_live_location() -> str:
    """
    Fetch live location city using an IP geolocation API.
    Returns the city name if successful, or defaults to 'shanghai'.
    """
    try:
        res = requests.get("http://ipinfo.io/json")
        data = res.json()
        city = data.get("city", "shanghai")
        return city.lower()  # lowercase to match API formatting
    except Exception as e:
        st.error(f"❌ Error fetching live location: {str(e)}")
        return "shanghai"


def get_pollution_data(city: str = None):
    """
    Fetch air pollution data from the WAQI API for the given city.
    If no city is provided, fetch the live location.
    Returns a dict with AQI, PM values, city, and time if successful.
    """
    if not city:
        city = get_live_location()
    url = f"http://api.waqi.info/feed/{city}/?token={POLLUTION_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("status") == "ok":
            iaqi = data["data"].get("iaqi", {})
            # Extract particulate matter values (e.g. pm25, pm10)
            pm_values = {key: iaqi[key]["v"] for key in iaqi if key.startswith("pm") and "v" in iaqi[key]}
            aqi = data["data"].get("aqi", "N/A")
            city_name = data["data"]["city"]["name"]
            time_str = data["data"]["time"]["s"]
            return {
                "aqi": aqi,
                "pm_values": pm_values,
                "city": city_name,
                "time": time_str
            }
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error fetching pollution data: {str(e)}")
        return None


# --------------------------
# Email Reminder Functions (Single & Multiple Tablets)
# --------------------------

def send_tablet_reminder(med_name: str, med_time: str, pollution_data: dict) -> bool:
    """
    Send a single tablet reminder email including current pollution info.
    """
    if pollution_data:
        aqi = pollution_data["aqi"]
        pm_values = pollution_data["pm_values"]
        pm_text = ", ".join([f"{k.upper()}: {v}" for k, v in pm_values.items()])
        city = pollution_data["city"]
        pol_time = pollution_data["time"]
    else:
        aqi = "N/A"
        pm_text = "N/A"
        city = "Unknown"
        pol_time = "N/A"

    email_subject = f"Medication Reminder for {med_name}"
    email_body = (
        f"Hello,\n\n"
        f"This is your reminder to take your medicine: {med_name} at {med_time}.\n\n"
        f"Pollution Alert (as of {pol_time}):\n"
        f"AQI in {city} is {aqi}.\n"
        f"PM levels: {pm_text}\n\n"
        "Stay healthy and take care!\n\n"
        "Your Lung Health Specialist Agent"
    )
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = email_subject
        msg.attach(MIMEText(email_body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email reminder: {str(e)}")
        return False


def send_tablet_reminder_multi(tablets: List[Tuple[str, str]], pollution_data: dict) -> bool:
    """
    Send a single email reminder listing multiple tablets and their times,
    along with the current pollution information.
    tablets: list of (tablet_name, time_str) tuples.
    """
    if pollution_data:
        aqi = pollution_data["aqi"]
        pm_values = pollution_data["pm_values"]
        pm_text = ", ".join([f"{k.upper()}: {v}" for k, v in pm_values.items()])
        city = pollution_data["city"]
        pol_time = pollution_data["time"]
    else:
        aqi = "N/A"
        pm_text = "N/A"
        city = "Unknown"
        pol_time = "N/A"

    email_subject = "Medication Reminder for Your Tablets"
    tablets_info = "\n".join([f"- {name} at {time}" for name, time in tablets])
    email_body = (
        f"Hello,\n\n"
        f"This is your reminder to take the following medicines:\n{tablets_info}\n\n"
        f"Pollution Alert (as of {pol_time}):\n"
        f"AQI in {city} is {aqi}.\n"
        f"PM levels: {pm_text}\n\n"
        "Please take your medication as scheduled and stay safe!\n\n"
        "Your Lung Health Specialist Agent"
    )
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = email_subject
        msg.attach(MIMEText(email_body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email reminder: {str(e)}")
        return False


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

st.title("🤧 Lung Pulmonology Specialist Agent")

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
    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input("Custom domains (comma-separated)",
                                           value=",".join(default_domains),
                                           help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org")
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Search Configuration
st.sidebar.header("🎯 Search Configuration")
st.session_state.similarity_threshold = st.sidebar.slider("Document Similarity Threshold", min_value=0.0, max_value=1.0,
                                                          value=0.7,
                                                          help="Lower values will return more documents but might be less relevant. Higher values are more strict.")

# Additional Features: Pollution Alert & Tablet Reminder
st.sidebar.header("🔔 Additional Features")
# Set the pollution alert threshold (AQI level above which reminders should be sent automatically)
pollution_threshold = st.sidebar.number_input("Pollution Alert Threshold (AQI)", min_value=0, max_value=500, value=100,
                                              step=1)
# Checkbox to enable tablet reminders in alerts
tablet_reminder_flag = st.sidebar.checkbox("Enable Tablet Reminders", value=False)


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
    """Initialize a query rewriting agent specialized for pulmonology."""
    return Agent(
        name="Query Rewriter",
        model=Gemini(id="gemini-exp-1206"),
        instructions=(
            "You are an expert at reformulating questions with a focus on lung health 🫁, diet, and respiratory care. "
            "Analyze the user's question, rewrite it to be more specific (e.g., expand 'COPD' to 'Chronic Obstructive Pulmonary Disease'), "
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
            "You are a web search expert focusing on lung health and nutrition 🥦. Search the web for up-to-date, reliable information "
            "on lung health, respiratory diets, and medical guidelines. Summarize the most relevant information and include sources."
        ),
        show_tool_calls=True,
        markdown=False
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent as a lung pulmonology specialist."""
    return Agent(
        name="Pulmonology Specialist Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions=(
            "Hello 👋, I’m your friendly Lung Health Specialist! I’m here to help you with personalized lung health advice, including diet plans, "
            "lifestyle tips, and clarifications on your respiratory concerns. "
            "If the current Air Quality Index (AQI) is high, I'll alert you about pollution levels. "
            "If you need a tablet reminder, I'll also remind you to take your medication. "
            "I always recommend consulting a healthcare professional for personalized care. "
            "When given context from documents, focus on extracting evidence-based information. "
            "When provided with web search results, indicate the source and synthesize the information in clear, plain text. "
            "Let's chat and work together to improve your lung health! 💪"
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

    # Create Tabs: Chat, Air Pollution Alert, and Medicine Schedule
    tabs = st.tabs(["Chat", "Air Pollution Alert", "Medicine Schedule"])

    # --------------------------
    # Tab 1: Chat Interface
    # --------------------------
    with tabs[0]:
        # Chat Interface: Two columns for chat input and search toggle
        chat_col, toggle_col = st.columns([0.9, 0.1])
        with chat_col:
            prompt = st.chat_input("Hello! How can I help you with your lung health today? 😊")
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
            if (
                    st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
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

            # Build additional context based on sidebar options
            additional_context = ""
            if pollution_threshold < 100:  # if manually set low, show an alert
                additional_context += f"\n⚠️ Pollution Alert: The current threshold is {pollution_threshold}."
            if tablet_reminder_flag:
                additional_context += "\n💊 Reminder: Don't forget your prescribed lung medication tablet."

            # Step 4: Generate response using the Pulmonology Specialist Agent
            with st.spinner("🤖 Generating your personalized advice..."):
                try:
                    rag_agent = get_rag_agent()
                    if context or additional_context:
                        full_prompt = (
                            f"Context: {context}{additional_context}\n\n"
                            f"User Question: {prompt}\n"
                            f"Revised Question: {rewritten_query}\n\n"
                            "Please provide a comprehensive, personalized lung health and diet plan in plain text with icons and emojis. "
                            "Greet the user warmly, clarify any doubts, and offer tailored advice. Always advise consulting a healthcare professional."
                        )
                    else:
                        full_prompt = (
                            f"User Question: {prompt}\n"
                            f"Revised Question: {rewritten_query}\n\n"
                            "Please provide personalized lung health and diet advice in plain text with icons and emojis. "
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
    # Tab 2: Air Pollution Alert
    # --------------------------
    with tabs[1]:
        st.header("🌫️ Air Pollution Alert")
        if st.button("Get Current Pollution Data"):
            with st.spinner("Fetching pollution data..."):
                # Fetch pollution data using live location
                pollution_data = get_pollution_data()
                if pollution_data:
                    st.success("✅ Pollution data retrieved!")
                    st.write(f"**City:** {pollution_data['city']}")
                    st.write(f"**AQI:** {pollution_data['aqi']}")
                    st.write("**PM Levels:**")
                    for pm, value in pollution_data["pm_values"].items():
                        st.write(f"- {pm.upper()}: {value}")
                    st.write(f"**Time:** {pollution_data['time']}")
                else:
                    st.error("Failed to retrieve pollution data.")

    # --------------------------
    # Tab 3: Medicine Schedule (Tablet Reminder)
    # --------------------------
    with tabs[2]:
        st.header("💊 Medicine Schedule")
        st.write("Enter your tablets (one per line) in the format: `Tablet Name, HH:MM`")
        tablets_input = st.text_area("Tablets", placeholder="Ventolin, 08:00\nSymbicort, 12:00")

        # Button to manually set reminders (optional individual sending)
        if st.button("Set Reminders Manually"):
            if not (tablets_input and GMAIL_USER and GMAIL_APP_PASSWORD and RECIPIENT_EMAIL):
                st.error("Please fill in all fields and ensure email credentials are set.")
            else:
                # Parse input into list of (tablet, time) tuples
                tablets = []
                for line in tablets_input.strip().splitlines():
                    if ',' in line:
                        parts = line.split(',')
                        tablet = parts[0].strip()
                        time_str = parts[1].strip()
                        tablets.append((tablet, time_str))
                if tablets:
                    with st.spinner("Fetching current pollution data..."):
                        pollution_data = get_pollution_data()
                    with st.spinner("Sending reminder email..."):
                        if send_tablet_reminder_multi(tablets, pollution_data):
                            st.success("✅ Reminder email sent!")
                        else:
                            st.error("Failed to send reminder email.")
                else:
                    st.error("No valid tablet entries found.")

        # Button to check pollution and automatically send reminders if AQI exceeds threshold
        if st.button("Check Pollution & Send Reminders Automatically"):
            if not (tablets_input and GMAIL_USER and GMAIL_APP_PASSWORD and RECIPIENT_EMAIL):
                st.error("Please fill in all fields and ensure email credentials are set.")
            else:
                with st.spinner("Fetching current pollution data..."):
                    pollution_data = get_pollution_data()
                try:
                    current_aqi = float(pollution_data.get("aqi", 0))
                except Exception:
                    current_aqi = 0
                if current_aqi >= pollution_threshold:
                    # Parse tablets as before
                    tablets = []
                    for line in tablets_input.strip().splitlines():
                        if ',' in line:
                            parts = line.split(',')
                            tablet = parts[0].strip()
                            time_str = parts[1].strip()
                            tablets.append((tablet, time_str))
                    if tablets:
                        with st.spinner("Sending automatic reminder email..."):
                            if send_tablet_reminder_multi(tablets, pollution_data):
                                st.success(
                                    f"✅ Automatic reminder sent! (Current AQI: {current_aqi} ≥ Threshold: {pollution_threshold})")
                            else:
                                st.error("Failed to send automatic reminder email.")
                    else:
                        st.error("No valid tablet entries found.")
                else:
                    st.info(
                        f"Current AQI ({current_aqi}) is below the threshold ({pollution_threshold}). No reminder sent.")

else:
    st.warning("⚠️ Please enter your Google API Key to continue")
