import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import streamlit as st
import chromadb
import os
from chromadb.config import Settings

# --- Configuration ---
# Choose your models (consider smaller versions for local use)
# LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1" # Example: Needs significant resources
LLM_MODEL_ID = "gpt2" # Using GPT-2 for easier local testing; substitute with Mistral/Llama if you have resources
TEXT_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5" # Good open-source text embedding model
IMAGE_EMBEDDING_MODEL_ID = "openai/clip-vit-base-patch32" # Standard CLIP model

# ChromaDB Configuration
CHROMA_PATH = os.path.join("data", "chroma_db")
TEXT_COLLECTION_NAME = "text_documents"
IMAGE_COLLECTION_NAME = "image_documents"

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {DEVICE} ---")

# --- Model Loading (Cached) ---

@st.cache_resource
def get_llm_pipeline(model_id=LLM_MODEL_ID):
    """Loads and caches the LLM pipeline."""
    print(f"Loading LLM: {model_id}...")
    # Add quantization config here if needed (e.g., load_in_8bit=True)
    # Requires bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # load_in_8bit=True, # Uncomment if using bitsandbytes and want 8-bit
        device_map='auto' # Automatically use GPU if available
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use 'text-generation' pipeline
    # Note: For instruct models, prompt formatting is crucial. GPT-2 is simpler.
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # torch_dtype=torch.float16, # Use float16 for faster inference if GPU supports it
        max_new_tokens=256, # Limit generated tokens
        device=0 if DEVICE == 'cuda' else -1 # pipeline device parameter: 0 for cuda:0, -1 for cpu
    )
    print("LLM loaded.")
    return llm_pipeline

@st.cache_resource
def get_text_embedding_model(model_id=TEXT_EMBEDDING_MODEL_ID):
    """Loads and caches the text embedding model."""
    print(f"Loading Text Embedding Model: {model_id}...")
    model = SentenceTransformer(model_id, device=DEVICE)
    print("Text Embedding Model loaded.")
    return model

@st.cache_resource
def get_image_embedding_models(model_id=IMAGE_EMBEDDING_MODEL_ID):
    """Loads and caches the CLIP model and processor."""
    print(f"Loading Image Embedding Model: {model_id}...")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(DEVICE)
    print("Image Embedding Model loaded.")
    return processor, model

@st.cache_resource
def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    print(f"Initializing ChromaDB client at: {CHROMA_PATH}")
    # Ensure the directory exists
    os.makedirs(CHROMA_PATH, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False) # Disable telemetry
    )
    print("ChromaDB client initialized.")
    return client

def get_chroma_collections(client):
    """Gets or creates the ChromaDB collections."""
    text_collection = client.get_or_create_collection(TEXT_COLLECTION_NAME)
    image_collection = client.get_or_create_collection(IMAGE_COLLECTION_NAME)
    print(f"Using collections: '{TEXT_COLLECTION_NAME}', '{IMAGE_COLLECTION_NAME}'")
    return text_collection, image_collection

# --- Load models immediately on import (or lazily if preferred) ---
# llm_pipeline = get_llm_pipeline() # Load lazily inside workflow if memory is tight
text_embedding_model = get_text_embedding_model()
clip_processor, clip_model = get_image_embedding_models()
chroma_client = get_chroma_client()
text_collection, image_collection = get_chroma_collections(chroma_client)