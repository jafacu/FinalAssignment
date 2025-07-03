
import streamlit as st
import base64
import tempfile
import chromadb
from pathlib import Path
from datetime import datetime
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter
import re

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice



# --- Background ---
def set_background_from_local_image(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



# --- Custom Styling ---
def apply_custom_styling():
    st.markdown(
        """
        <style>
        /* Set the background image */
        body {
            background-image: url("background.jpg");
            background-size: cover;
            background-attachment: fixed;
        }

        /* Center and style main content */
        .main > div {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
            max-width: 900px;
            margin: auto;
        }

        /* Header styling */
        .stApp h1, .stApp h2, .stApp h3 {
            text-align: center;
            color: #ffffff;
        }

        /* Tab labels */
        .stTabs [data-baseweb="tab"] {
            color: #ffffff;
            font-weight: bold;
        }

        /* Input boxes, buttons, etc. */
        .stTextInput input, .stTextArea textarea, .stButton button, .stFileUploader {
            color: #ffffff !important;
            background-color: #00000 !important;
            border-radius: 8px;
        }

        /* Button hover */
        .stButton button:hover {
            background-color: #00000 !important;
            color: #ffffff !important;
        }

        /* File uploader tweaks */
        .stFileUploader {
            background-color: rgba(0,0,0,0);
        }
        </style>
        """,
        unsafe_allow_html=True
    )



# --- Tab 1: Upload and Store ---

# -- Function support --
# - Conversion -
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts, backend=DoclingParseV2DocumentBackend)
        })
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")

# - ChromaDB -
def setup_documents():
    client = chromadb.Client()
    try:
        return client.get_collection(name="documents")
    except:
        return client.create_collection(name="documents")

def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_text(text)
    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}
    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection
    collection = add_text_to_chromadb.collections[collection_name]
    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()
        metadata = { "filename": filename, "chunk_index": i, "chunk_size": len(chunk) }
        collection.add(embeddings=[embedding], documents=[chunk], metadatas=[metadata], ids=[f"{filename}_chunk_{i}"])
    return collection


# -- Tab 1 Main Function --
def upload_and_store():
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True)
    if st.button("Convert & Store"):
        if uploaded_files and len(uploaded_files) <= 5:
            with st.spinner("Converting and storing files... 📂"):
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    try:
                        text = convert_to_markdown(tmp_path)
                        st.session_state.collection = add_text_to_chromadb(text, file.name)
                        st.session_state.converted_docs.append({ "filename": file.name, "content": text })
                        st.success(f"Added {file.name}")
                    except Exception as e:
                        st.error(f"{file.name} error: {e}")
        elif uploaded_files:
            st.error("Max 5 files.")
        else:
            st.warning("No files uploaded.")



# --- Tab 2: Q&A ---

# -- Function support --
# - Q&A -
def get_answer(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic."
    context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context:
{context}

Question: {question}

Instructions: Answer ONLY from the context above. If not present, say 'I don't know.'

Answer:"""
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    return response[0]['generated_text'].strip()

# - Search History -
def add_to_search_history(question, answer, source):
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    if 'search_history' in st.session_state and st.session_state.search_history:
        st.subheader("🕒 Search History")
        for entry in st.session_state.search_history:
            with st.expander(f"{entry['timestamp']} - {entry['question']}"):
                st.write("**Answer:**", entry['answer'])
                st.write("**Source:**", entry['source'])


# -- Tab 2 Main Function --
def q_and_a():
    if st.session_state.converted_docs:
        question = st.text_input("Ask something:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Thinking... 🤖"):
                    answer = get_answer(st.session_state.collection, question)
                st.success("Done!")
                st.write("**Answer:**")
                st.write(answer)
                add_to_search_history(question, answer, "ChromaDB")
        show_search_history()
    else:
        st.info("Please upload documents first.")



# --- Tab 3: Document Management ---

# -- Tab 3 Main Function --
def document_manager():
    if not st.session_state.get("converted_docs"):
        st.info("No documents uploaded yet.")
        return
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        with col2:
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                deleted_filename = st.session_state.converted_docs[i]["filename"]
                st.session_state.converted_docs.pop(i)
                existing_ids = st.session_state.collection.get()["ids"]
                matching_ids = [id for id in existing_ids if id.startswith(f"{deleted_filename}_chunk_")]
                if matching_ids:
                    st.session_state.collection.delete(ids=matching_ids)
                st.rerun()
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()



# --- Tab 4: Document Summarization ---

# -- Function support --
# - Summary -
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    try:
        return summarizer(text[:1024])[0]['summary_text']
    except Exception as e:
        return "Summary failed."
    
# -- Tab 4 Main Function --
def summarize_documents():
    if st.session_state.converted_docs:
        for doc in st.session_state.converted_docs:
            with st.expander(f"Summary of {doc['filename']}"):
                st.write(summarize_text(doc["content"]))
    else:
        st.info("Upload documents first.")



# --- Tab 5: Keywords & Topics ---

# -- Function support --
# - Topics -
def extract_keywords(text, top_n=10):
    tokens = re.findall(r'\b\w{5,}\b', text.lower())
    common = Counter(tokens).most_common(top_n)
    return [word for word, _ in common]


# -- Tab 5 Main Function --
def topics():
    if st.session_state.converted_docs:
        for doc in st.session_state.converted_docs:
            with st.expander(f"Topics in {doc['filename']}"):
                keywords = extract_keywords(doc["content"])
                st.write(", ".join(keywords))
    else:
        st.info("Upload documents first.")




# --- Main ---

apply_custom_styling()
def main():
    set_background_from_local_image("background.jpg")
    st.set_page_config(page_title="Smart Document App")
    st.title("📂🗨️ FileTalk App")

    if 'collection' not in st.session_state:
        st.session_state.collection = setup_documents()
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload", "❓ Q&A", "📋 Manage", "🧾 Summarize", "🔎 Topics"])

    with tab1:
        st.header("⬆️ Upload & Store Documents")
        upload_and_store()

    with tab2:
        st.header("❓ Ask Questions")
        q_and_a()

    with tab3:
        st.header("📋 Manage Documents")
        document_manager()

    with tab4:
        st.header("🧾 Document Summaries")
        summarize_documents()

    with tab5:
        st.header("🔎 Keywords & Topics")
        topics()
    
    # Footer
    st.markdown("---")
    st.markdown("*Designed by Amos Facundo • Built with Streamlit • Powered by AI*")

if __name__ == "__main__":
    main()
