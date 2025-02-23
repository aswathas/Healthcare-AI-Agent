import os
import streamlit as st
import google.generativeai as genai
import PyPDF2
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------- Sidebar for Gemini API Key ----------------
with st.sidebar:
    st.header("API Settings")
    gemini_key_input = st.text_input("Enter your Gemini API Key", type="password")

# Use the API key from the sidebar if provided; otherwise, fallback to secrets or environment variables.
API_KEY = gemini_key_input or st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "AIzaSyAP7VOyDFYt97eVE5RMKLIJrlqzl9GXpXo"))
genai.configure(api_key=API_KEY)

# ---------------- Custom LLM Wrapper for Gemini Flash Thinking Model ----------------
class GeminiFlashLLM(LLM):
    # Fields declared as Pydantic attributes.
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21"
    generation_config: dict = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        # Removed "max_input_tokens" because it is not a recognized field.
        "max_output_tokens": 65536,
        "response_mime_type": "text/plain",
    }

    @property
    def _llm_type(self) -> str:
        return "gemini_flash"

    def _call(self, prompt: str, stop=None) -> str:
        # Create a new chat session with an empty history for report generation.
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text if response and hasattr(response, "text") else "No output received."

# ---------------- Function for Chat Mode using Gemini Flash Thinking Model ----------------
def get_chat_response(user_message: str) -> str:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            # Removed "max_input_tokens" here as well.
            "max_output_tokens": 65536,
            "response_mime_type": "text/plain",
        },
    )
    # Start a new chat session with the existing history.
    chat_session = model.start_chat(history=st.session_state.chat_history)
    response = chat_session.send_message(user_message)
    # Append the user's message and the assistant's reply to the chat history.
    st.session_state.chat_history.append({"author": "user", "content": user_message})
    st.session_state.chat_history.append({"author": "assistant", "content": response.text})
    return response.text

# ---------------- Initialize Session State for Chat History ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "author": "system",
            "content": "You are a helpful pulmonologist specialist doctor. Provide clear medical advice along with dietary recommendations when appropriate."
        }
    ]

# ---------------- Create Two Tabs: Report Generator and Chat ----------------
tab1, tab2 = st.tabs(["Generate Report", "Chat with Doctor"])

# ---------------- Tab 1: Report Generator ----------------
with tab1:
    st.title("Pulmonologist AI Doctor Agent - Report Generator")
    st.write(
        "Upload a patient document (TXT or PDF) to generate a detailed pulmonology report with observations, possible diagnoses, treatment recommendations, dietary suggestions, and further testing suggestions."
    )

    uploaded_file = st.file_uploader("Upload Patient Document (TXT or PDF)", type=["txt", "pdf"])
    document_text = ""

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".txt"):
            try:
                document_text = uploaded_file.read().decode("utf-8")
                st.subheader("Uploaded Document (Text)")
                st.text_area("Patient Document", document_text, height=200)
            except Exception as e:
                st.error(f"Error reading the text file: {e}")
        elif file_name.endswith(".pdf"):
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pages_text = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                document_text = "\n".join(pages_text)
                st.subheader("Uploaded Document (PDF)")
                st.text_area("Patient Document", document_text, height=200)
            except Exception as e:
                st.error(f"Error reading the PDF file: {e}")

    if document_text:
        prompt_template = """
You are a pulmonologist specialist doctor. Analyze the following patient document and generate a detailed report including:
- Observations from the document
- Possible diagnoses
- Treatment recommendations
- Dietary recommendations
- Suggestions for further testing

Patient Document:
{document_text}

Please provide a clear, concise, and medically-informed report.
"""
        prompt = PromptTemplate(input_variables=["document_text"], template=prompt_template)
        gemini_llm = GeminiFlashLLM()  # Use the Gemini Flash Thinking model for report generation.
        llm_chain = LLMChain(llm=gemini_llm, prompt=prompt)

        if st.button("Generate Report"):
            st.write("### Generating Report...")
            try:
                report = llm_chain.run(document_text=document_text)
                st.subheader("Generated Pulmonology Report")
                st.write(report)
            except Exception as e:
                st.error(f"Error generating report: {e}")

# ---------------- Tab 2: Chat with Doctor ----------------
with tab2:
    st.title("Chat with Your Pulmonologist Specialist Doctor")
    st.write("Ask questions or discuss your concerns. The doctor will provide medical advice and dietary recommendations.")

    # Display chat history.
    chat_placeholder = st.empty()
    for msg in st.session_state.chat_history:
        if msg["author"] == "system":
            continue  # Optionally hide system messages.
        if msg["author"] == "user":
            chat_placeholder.markdown(f"**You:** {msg['content']}")
        else:
            chat_placeholder.markdown(f"**Doctor:** {msg['content']}")

    # Chat input area.
    user_input = st.text_input("Your message:", key="chat_input")
    if st.button("Send", key="send_button") and user_input:
        get_chat_response(user_input)
        # Update the chat display.
        chat_placeholder.empty()
        for msg in st.session_state.chat_history:
            if msg["author"] == "system":
                continue
            if msg["author"] == "user":
                chat_placeholder.markdown(f"**You:** {msg['content']}")
            else:
                chat_placeholder.markdown(f"**Doctor:** {msg['content']}")
