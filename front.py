import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your custom model - this appears correct in original code
from model import MedicalLanguageModel

# Properly configure event loop (moved to before any torch imports)
import asyncio
import sys
if sys.platform == "darwin":  # Fix event loop issue on macOS
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Streamlit cache to load model only once
@st.cache_resource
def load_model():
    """Load the model only once and cache it for future use"""
    st.write("🔄 Loading medical model... (This happens only once!)")
    try:
        model_instance = MedicalLanguageModel("epfl-llm/meditron-7b", device="cpu")
        st.write("✅ Model loaded successfully!")
        return model_instance.tokenizer, model_instance.model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Streamlit UI
st.title("🩺 Meditron Medical AI Chatbot")
st.write("Enter a medical query and get AI-generated responses.")

# User input
user_input = st.text_area("Enter your question:", placeholder="e.g., What are the symptoms of diabetes?")
temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
max_length = st.slider("Max Length", 50, 500, 200)

# Load model on app startup
tokenizer, model = load_model()

if st.button("Generate Response"):
    if model is None or tokenizer is None:
        st.error("🚨 Model failed to load. Please restart the application.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Process input
                inputs = tokenizer(user_input, return_tensors="pt")
                
                # Ensure tensors are on CPU
                inputs = {key: val.to("cpu") for key, val in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode and display response
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                st.write("### Response:")
                st.success(response)
            except Exception as e:
                st.error(f"🚨 Error generating response: {str(e)}")
                st.error("For detailed traceback, check your terminal")