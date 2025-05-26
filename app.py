import streamlit as st
from PIL import Image
import torch
import logging
import time # Import time for simulated delays (can be removed later)
from transformers import ViltProcessor, ViltForQuestionAnswering # Import ViLT components

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting Streamlit VQA application...")

st.title("Visual Question Answering (VQA)")

st.write("Upload an image and ask a question about it!")

# --- VQA Model and Processor Setup ---
# Load the pre-trained ViLT processor and model
@st.cache_resource # Cache the processor and model
def get_vqa_model():
    logging.info("Loading pre-trained ViLT processor and model...")
    # Using a fine-tuned ViLT model for VQA
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    logging.info("ViLT processor and model loaded.")
    return processor, model

# -------------------------------------


# Image Upload
logging.info("Setting up image uploader...")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file is not None:
    logging.info(f"Image uploaded: {uploaded_file.name}")
    st.info("Image uploaded successfully.")
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_container_width=True)

# Question Input
logging.info("Setting up question input field...")
question = st.text_input("Ask a question about the image:")

# VQA Inference Button
logging.info("Setting up Get Answer button...")
if st.button("Get Answer"):
    logging.info("Get Answer button clicked.")
    if image is not None and question:
        st.write("Processing...")
        
        # Create a status container to update messages
        status_text = st.empty()

        # --- VQA Inference with Pre-trained Model ---
        
        # Load the processor and model
        status_text.info("Loading VQA model components...")
        processor, model = get_vqa_model()
        status_text.success("VQA model components loaded.")
        logging.info("VQA model components retrieved for inference.")

        # Prepare inputs using the ViLT processor
        status_text.info("Step 1: Preparing image and question inputs...")
        logging.info("Starting input preparation...")
        try:
            # The ViLT processor handles both image preprocessing and text tokenization
            inputs = processor(image, question, return_tensors="pt")
            status_text.success("Step 1: Image and question inputs prepared.")
            logging.info("Image and question inputs prepared.")
        except Exception as e:
            logging.error(f"Error preparing inputs: {e}")
            status_text.error(f"Error preparing inputs: {e}")
            st.stop()

        # Run the VQA model for inference
        status_text.info("Step 2: Running VQA model...")
        logging.info("Starting VQA model inference...")
        try:
            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Get the predicted answer
            # ViLT outputs logits over the possible answers
            logits = outputs.logits
            idx = logits.argmax(-1).item() # Get the index of the highest logit
            predicted_answer = model.config.id2label[idx] # Map the index to the answer string
            
            # Simulate inference time if needed for demonstration (can remove)
            time.sleep(2)
            # ------------------------------------------------------------
            
            status_text.empty() # Clear the status message
            st.success(f"Answer: {predicted_answer}")
            logging.info(f"VQA inference complete. Predicted answer: {predicted_answer}")
        except Exception as e:
            logging.error(f"Error during VQA inference: {e}")
            status_text.error(f"Error during VQA inference: {e}")
            st.stop()

        logging.info("VQA inference process finished.")


    elif image is None and question:
        logging.warning("Get Answer clicked without image.")
        st.warning("Please upload an image.")
    elif image is not None and not question:
        logging.warning("Get Answer clicked without question.")
        st.warning("Please enter a question.")
    else:
        logging.info("Get Answer clicked without image and question.")
        st.info("Please upload an image and enter a question to get an answer.")

# Add instructions or notes
st.markdown("**Note:** This app uses a pre-trained ViLT model from Hugging Face for VQA. Ensure you have the `transformers` library installed.")

logging.info("Streamlit app initialization complete.") 