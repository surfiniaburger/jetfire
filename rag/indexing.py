import os
from utils import (
    text_embedding_model,
    clip_processor,
    clip_model,
    text_collection,
    image_collection,
    DEVICE,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME
)
from PIL import Image
import torch
import uuid

# --- Configuration ---
SOURCE_DATA_DIR = "sample_data" # Directory containing files to index
TEXT_CHUNK_SIZE = 512 # Size of text chunks (adjust as needed)

# --- Helper Functions ---
def chunk_text(text, chunk_size):
    """Simple text chunking."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# --- Indexing Logic ---
def index_data(source_dir):
    print(f"Starting indexing process for directory: {source_dir}")
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    text_docs = []
    text_metadatas = []
    text_ids = []
    image_files = []
    image_ids = []

    # 1. Discover files
    for filename in os.listdir(source_dir):
        filepath = os.path.join(source_dir, filename)
        if filename.lower().endswith(".txt"):
            print(f"Found text file: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = chunk_text(content, TEXT_CHUNK_SIZE)
                for i, chunk in enumerate(chunks):
                    doc_id = f"{filename}_{i}"
                    text_docs.append(chunk)
                    text_metadatas.append({"source": filename, "chunk_index": i})
                    text_ids.append(doc_id)
            except Exception as e:
                print(f"Error reading text file {filename}: {e}")

        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            print(f"Found image file: {filename}")
            image_files.append(filepath)
            image_ids.append(str(uuid.uuid4())) # Use UUID for image IDs
        else:
            print(f"Skipping unsupported file: {filename}")

    # 2. Process and index text documents
    if text_docs:
        print(f"Generating embeddings for {len(text_docs)} text chunks...")
        text_embeddings = text_embedding_model.encode(text_docs, convert_to_tensor=False, show_progress_bar=True)
        print(f"Adding {len(text_docs)} text chunks to ChromaDB collection '{TEXT_COLLECTION_NAME}'...")
        try:
            text_collection.add(
                embeddings=text_embeddings.tolist(),
                documents=text_docs,
                metadatas=text_metadatas,
                ids=text_ids
            )
            print("Text indexing complete.")
        except Exception as e:
             print(f"Error adding text to ChromaDB: {e}")
             # Consider handling potential duplicate IDs if re-indexing
             # E.g., use upsert=True if supported and desired
             # text_collection.upsert(...)
    else:
        print("No text documents found to index.")

    # 3. Process and index images
    if image_files:
        print(f"Generating embeddings for {len(image_files)} images...")
        image_embeddings_list = []
        valid_image_files = []
        valid_image_ids = []
        valid_metadatas = []

        for img_path, img_id in zip(image_files, image_ids):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = clip_processor(text=None, images=image, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
                image_embeddings_list.append(image_features.cpu().numpy().squeeze())
                valid_image_files.append(img_path) # Store path as metadata
                valid_image_ids.append(img_id)
                valid_metadatas.append({"source": img_path}) # Use path as source
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        if image_embeddings_list:
            print(f"Adding {len(valid_image_ids)} image embeddings to ChromaDB collection '{IMAGE_COLLECTION_NAME}'...")
            try:
                 # ChromaDB expects embeddings as a list of lists/arrays
                image_collection.add(
                    embeddings=[emb.tolist() for emb in image_embeddings_list],
                     # For images, 'documents' can be paths or IDs; paths are useful
                    documents=valid_image_files,
                    metadatas=valid_metadatas,
                    ids=valid_image_ids
                )
                print("Image indexing complete.")
            except Exception as e:
                print(f"Error adding images to ChromaDB: {e}")
                # Consider handling potential duplicate IDs if re-indexing
                # image_collection.upsert(...)
        else:
             print("No valid images processed for indexing.")

    else:
        print("No image files found to index.")

    print("Indexing process finished.")

if __name__ == "__main__":
    # Create sample data directory if it doesn't exist
    if not os.path.exists(SOURCE_DATA_DIR):
        os.makedirs(SOURCE_DATA_DIR)
        # Create dummy files for testing
        with open(os.path.join(SOURCE_DATA_DIR, "doc1.txt"), "w") as f:
            f.write("This is the first sample document about cats. Cats are furry companions.")
        with open(os.path.join(SOURCE_DATA_DIR, "doc2.txt"), "w") as f:
            f.write("The second document discusses dogs. Dogs are loyal animals often kept as pets.")
        # You'll need to add actual image files (e.g., image1.jpg) to the sample_data folder yourself
        print(f"Created sample directory '{SOURCE_DATA_DIR}'. Add text and image files there.")

    index_data(SOURCE_DATA_DIR)