import langgraph as lg
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional
import operator # For state updates
from utils import (
    text_embedding_model,
    clip_processor,
    clip_model,
    text_collection,
    image_collection,
    DEVICE,
    get_llm_pipeline # Load LLM lazily here if preferred
)
from PIL import Image
import torch
import base64
import io

# --- State Definition ---
class RAGState(TypedDict):
    query_text: Optional[str]
    query_image_path: Optional[str] # Path to the temporarily saved image
    query_image_b64: Optional[str] # Base64 encoded image for display

    text_embedding: Optional[List[float]]
    image_embedding: Optional[List[float]]

    retrieved_text_docs: List[str]
    retrieved_image_paths: List[str] # Store paths of retrieved images

    generation: str
    error: Optional[str]


# --- Node Functions ---

def encode_query(state: RAGState) -> RAGState:
    """Encodes text query and/or image query."""
    print("--- Node: encode_query ---")
    text_emb = None
    image_emb = None
    error = None

    try:
        # Encode text if present
        if state.get("query_text"):
            print(f"Encoding text query: '{state['query_text'][:50]}...'")
            text_emb = text_embedding_model.encode(state["query_text"], convert_to_tensor=False).tolist()
            print(f"Text embedding generated (shape: {len(text_emb)})")

        # Encode image if present
        if state.get("query_image_path"):
            print(f"Encoding image query: {state['query_image_path']}")
            try:
                image = Image.open(state["query_image_path"]).convert("RGB")
                inputs = clip_processor(text=None, images=image, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
                image_emb = image_features.cpu().numpy().squeeze().tolist()
                print(f"Image embedding generated (shape: {len(image_emb)})")

                # Also encode image to base64 for potential display later
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG") # Or PNG
                img_str = base64.b64encode(buffered.getvalue()).decode()
                state["query_image_b64"] = img_str

            except Exception as e:
                print(f"Error encoding image: {e}")
                error = f"Error encoding image: {e}"

    except Exception as e:
        print(f"Error during encoding: {e}")
        error = f"Error during encoding: {e}"


    return {
        "text_embedding": text_emb,
        "image_embedding": image_emb,
        "error": error # Propagate error if any
    }

def retrieve_documents(state: RAGState) -> RAGState:
    """Performs hybrid retrieval from ChromaDB based on available embeddings."""
    print("--- Node: retrieve_documents ---")
    if state.get("error"): # Skip if previous step had error
         return {"retrieved_text_docs": [], "retrieved_image_paths": []}

    retrieved_texts = []
    retrieved_image_paths_from_text = []
    retrieved_image_paths_from_image = []
    final_image_paths = []
    n_results = 3 # Number of results to fetch for each modality

    try:
        # 1. Retrieve text documents based on text query embedding
        if state.get("text_embedding"):
            print(f"Querying text collection with text embedding...")
            results = text_collection.query(
                query_embeddings=[state["text_embedding"]],
                n_results=n_results,
                include=['documents', 'metadatas'] # Include documents and metadata
            )
            if results and results.get('documents') and results['documents'][0]:
                 retrieved_texts = results['documents'][0]
                 print(f"Retrieved {len(retrieved_texts)} text snippets.")
            else:
                 print("No text documents found for text query.")


        # 2. Retrieve images based on image query embedding (if available)
        if state.get("image_embedding"):
            print(f"Querying image collection with image embedding...")
            results = image_collection.query(
                query_embeddings=[state["image_embedding"]],
                n_results=n_results,
                 # Documents in image collection are paths, metadatas might have more info
                include=['documents', 'metadatas']
            )
            if results and results.get('documents') and results['documents'][0]:
                retrieved_image_paths_from_image = results['documents'][0]
                print(f"Retrieved {len(retrieved_image_paths_from_image)} images based on image query.")
            else:
                print("No images found for image query.")

        # Simple combination: Unique image paths from both queries
        # More sophisticated ranking could be added here (e.g., using scores if Chroma returns them)
        all_image_paths = retrieved_image_paths_from_text + retrieved_image_paths_from_image
        # Use dict to preserve order while getting unique paths
        final_image_paths = list(dict.fromkeys(all_image_paths))

    except Exception as e:
        print(f"Error during retrieval: {e}")
        state["error"] = f"Error during retrieval: {e}"


    return {
        "retrieved_text_docs": retrieved_texts,
        "retrieved_image_paths": final_image_paths,
    }


def format_context(state: RAGState) -> dict:
    """Formats retrieved documents and image info into a context string for the LLM."""
    print("--- Node: format_context (preparing LLM input) ---")
    if state.get("error"):
        return {"generation": "Error occurred before generation."}

    context = ""
    if state.get("retrieved_text_docs"):
        context += "Relevant text passages:\n"
        for i, doc in enumerate(state["retrieved_text_docs"]):
            context += f"{i+1}. {doc}\n\n"

    if state.get("retrieved_image_paths"):
        context += "Relevant images found (by path):\n"
        for i, path in enumerate(state["retrieved_image_paths"]):
            context += f"- {os.path.basename(path)}\n" # Just show filename

    # Basic check if any context was found
    if not context:
        context = "No relevant information found in the knowledge base.\n"

    # Construct the prompt
    query = state.get("query_text") or "Describe the provided image."
    if state.get("query_image_path") and not state.get("query_text"):
        query = f"Regarding the uploaded image ({os.path.basename(state['query_image_path'])}): Describe it or answer based on it using the context."
    elif state.get("query_image_path") and state.get("query_text"):
         query = f"Regarding the query '{state['query_text']}' and the uploaded image ({os.path.basename(state['query_image_path'])}):"


    # Simple prompt structure (adapt for specific instruct models like Mistral)
    # For Mistral Instruct: `<s>[INST] {prompt} [/INST]`
    prompt = f"""Based on the following context, answer the query.

Context:
{context}

Query: {query}

Answer:
"""
    print(f"Generated Prompt for LLM:\n{prompt}")
    return {"prompt": prompt} # Pass prompt to the next step

def generate_response(state: RAGState) -> dict:
    """Generates a response using the LLM based on the formatted prompt."""
    print("--- Node: generate_response ---")
    if state.get("error") or not state.get("prompt"):
        # If there was an error earlier, or no prompt was generated, return error message
        error_msg = state.get("error", "Skipping generation due to missing prompt.")
        print(error_msg)
        # Ensure the final state reflects the error encountered earlier or here
        return {"generation": error_msg, "error": state.get("error") or "Missing prompt"}


    prompt = state["prompt"]
    generation = "Failed to generate response." # Default fail message
    try:
        # Load LLM here if loading lazily
        llm_pipeline = get_llm_pipeline()
        print("Invoking LLM...")
        # Note: The pipeline might return the prompt + generation. Extract only the new part.
        # Adjust parameters like max_length, temperature as needed.
        results = llm_pipeline(prompt, max_new_tokens=150) # Adjust max_new_tokens
        if results and isinstance(results, list) and 'generated_text' in results[0]:
            generated_text = results[0]['generated_text']
            # Try to extract only the generated part after the prompt
            # This logic might need adjustment based on the specific LLM's output format
            answer_marker = "Answer:"
            marker_pos = generated_text.rfind(answer_marker) # Find the last occurrence
            if marker_pos != -1:
                 generation = generated_text[marker_pos + len(answer_marker):].strip()
            else:
                 # Fallback if "Answer:" marker isn't found in the expected place
                 # This might happen if the LLM doesn't follow the instruction perfectly
                 # Or if the prompt itself was part of the input sequence returned by pipeline
                 # Let's try splitting by the end of the original prompt
                 prompt_end_pos = generated_text.find(prompt)
                 if prompt_end_pos != -1:
                      generation = generated_text[prompt_end_pos + len(prompt):].strip()
                 else: # Raw output if markers fail
                      generation = generated_text

            print(f"LLM Generation successful: '{generation[:100]}...'")
        else:
            print(f"LLM pipeline returned unexpected result format: {results}")
            state["error"] = "LLM pipeline returned unexpected result format."


    except Exception as e:
        print(f"Error during LLM generation: {e}")
        state["error"] = f"Error during LLM generation: {e}"
        generation = f"Error during LLM generation: {e}"


    return {"generation": generation}


# --- Build the Graph ---
def build_rag_graph():
    print("Building LangGraph workflow...")
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("encode_query", encode_query)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("format_context", format_context)
    workflow.add_node("generate_response", generate_response)

    # Define edges
    workflow.set_entry_point("encode_query")
    workflow.add_edge("encode_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "format_context")
    workflow.add_edge("format_context", "generate_response")
    workflow.add_edge("generate_response", END) # End after generation

    # Compile the graph
    app = workflow.compile()
    print("LangGraph workflow built successfully.")
    return app

# Instantiate the graph
langgraph_app = build_rag_graph()