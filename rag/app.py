import streamlit as st
from rag_workflow import langgraph_app, RAGState # Import the compiled app and state
import os
from PIL import Image
import tempfile # To save uploaded image temporarily

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="Multimodal RAG with LangGraph")

st.title("📚 Multimodal RAG System 🖼️")
st.caption("Query using text and/or images with LangGraph, ChromaDB, and Open Source Models")

# --- Session State Initialization ---
# Store workflow results in session state to avoid rerunning on minor UI interactions
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# --- Input Area ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Text Query")
    query_text = st.text_area("Enter your text query here:", height=150)

with col2:
    st.subheader("Image Query")
    uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg", "webp"])

# --- Execution Button ---
if st.button("🚀 Generate Response"):
    if not query_text and not uploaded_image:
        st.warning("Please provide a text query or upload an image.")
    else:
        # Prepare inputs for LangGraph
        inputs = RAGState(
            query_text=query_text if query_text else None,
            query_image_path=None,
            text_embedding=None,
            image_embedding=None,
            retrieved_text_docs=[],
            retrieved_image_paths=[],
            generation="",
            error=None,
            query_image_b64=None # Initialize
        )

        temp_image_path = None # Keep track of temp file path

        # Handle image upload: save temporarily
        if uploaded_image is not None:
            try:
                # Create a temporary file to save the uploaded image
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_image.getvalue())
                    temp_image_path = tmp_file.name # Get the path to the temporary file
                    inputs["query_image_path"] = temp_image_path
                st.image(uploaded_image, caption="Uploaded Image Query", width=200)
            except Exception as e:
                st.error(f"Error handling uploaded image: {e}")
                inputs["error"] = f"Error handling uploaded image: {e}" # Set error state

        # --- Invoke LangGraph Workflow ---
        if not inputs.get("error"): # Proceed only if no error during image handling
            final_state = None
            with st.spinner("Processing your query... (Encoding -> Retrieving -> Generating)"):
                try:
                    # Stream the events to see progress (optional but good for debugging)
                    # for event in langgraph_app.stream(inputs):
                    #     for key, value in event.items():
                    #         print(f"--- Event: {key} ---")
                    #         # print(value) # Print the state changes (can be verbose)
                    #     final_state = list(event.values())[-1] # Get the last state update

                    # Or just invoke and get the final state directly
                    final_state = langgraph_app.invoke(inputs)

                    st.session_state.last_result = final_state # Store result in session state

                except Exception as e:
                    st.error(f"An error occurred during the RAG workflow: {e}")
                    st.session_state.last_result = {"error": str(e), "generation": "Workflow failed."} # Store error state

        # --- Cleanup ---
        # Clean up the temporary image file after processing
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"Cleaned up temporary file: {temp_image_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_image_path}: {e}")

# --- Display Results ---
if st.session_state.last_result:
    result = st.session_state.last_result
    st.divider()
    st.subheader("✨ Generated Response")

    if result.get("error"):
        st.error(f"Workflow Error: {result.get('error')}")

    st.markdown(result.get("generation", "No response generated."))

    # Display retrieved items
    st.subheader("🔍 Retrieved Context")
    retrieved_texts = result.get("retrieved_text_docs", [])
    retrieved_images = result.get("retrieved_image_paths", [])

    if not retrieved_texts and not retrieved_images:
        st.info("No relevant context was retrieved from the database.")
    else:
        if retrieved_texts:
            with st.expander("Retrieved Text Snippets", expanded=False):
                for i, doc in enumerate(retrieved_texts):
                    st.markdown(f"**Snippet {i+1}:**")
                    st.markdown(f"> {doc}")
                    st.markdown("---") # Separator

        if retrieved_images:
            with st.expander("Retrieved Images", expanded=True):
                 # Create columns for image display
                cols = st.columns(min(len(retrieved_images), 4)) # Display up to 4 images per row
                for i, img_path in enumerate(retrieved_images):
                    try:
                        # Check if the image file exists before trying to display it
                        if os.path.exists(img_path):
                            image = Image.open(img_path)
                            cols[i % 4].image(image, caption=f"{os.path.basename(img_path)}", use_column_width=True)
                        else:
                             cols[i % 4].warning(f"Image not found:\n{os.path.basename(img_path)}")
                    except Exception as e:
                        cols[i % 4].error(f"Error loading:\n{os.path.basename(img_path)}\n{e}")

# Add some info about the models being used
st.sidebar.title("Configuration Info")
st.sidebar.markdown(f"**LLM:** `{LLM_MODEL_ID}`")
st.sidebar.markdown(f"**Text Embeddings:** `{TEXT_EMBEDDING_MODEL_ID}`")
st.sidebar.markdown(f"**Image Embeddings:** `{IMAGE_EMBEDDING_MODEL_ID}`")
st.sidebar.markdown(f"**Vector DB:** ChromaDB @ `{CHROMA_PATH}`")
st.sidebar.markdown(f"**Compute Device:** `{DEVICE.upper()}`")