import streamlit as st
import torch
import torch.nn as nn
import os
import sys
import importlib.util
import subprocess
import numpy as np
import py3Dmol
import streamlit.components.v1 as components
import math

# ==========================================
# 1. SETUP & CACHING (Runs only once)
# ==========================================
@st.cache_resource
def load_model_and_tokenizer():
    """Clones the repo, patches imports, and loads the fine-tuned model."""
    base_dir = '/kaggle/working/caduceus_model' # Or a local path like './caduceus_model'
    
    # Clone repo if not exists
    if not os.path.exists(base_dir):
        subprocess.run(['git', 'clone', 'https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16', base_dir])

    # Patch imports (Crucial for Streamlit environment)
    for file in ['modeling_caduceus.py', 'modeling_rcps.py']:
        path = os.path.join(base_dir, file)
        with open(path, 'r') as f:
            content = f.read()
        content = content.replace('from .configuration_caduceus', 'from configuration_caduceus')
        content = content.replace('from .modeling_rcps', 'from modeling_rcps')
        with open(path, 'w') as f:
            f.write(content)

    # Add to path and load modules
    sys.path.insert(0, base_dir)
    
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load_module('configuration_caduceus', f'{base_dir}/configuration_caduceus.py')
    load_module('modeling_rcps', f'{base_dir}/modeling_rcps.py')
    load_module('modeling_caduceus', f'{base_dir}/modeling_caduceus.py')
    load_module('tokenization_caduceus', f'{base_dir}/tokenization_caduceus.py')

    from configuration_caduceus import CaduceusConfig
    from modeling_caduceus import Caduceus
    from tokenization_caduceus import CaduceusTokenizer

    # Load Config & Tokenizer
    config = CaduceusConfig.from_pretrained(base_dir)
    tokenizer = CaduceusTokenizer.from_pretrained(base_dir)

    # Define Model Architecture
    class CaduceusClassifier(nn.Module):
        def __init__(self, config, num_labels=5):
            super().__init__()
            self.encoder = Caduceus(config)
            hidden_dim = config.d_model * 2 
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_labels)
            )

        def forward(self, input_ids, labels=None):
            outputs = self.encoder(input_ids=input_ids)
            pooled = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs[0].mean(dim=1)
            logits = self.classifier(pooled)
            return {'loss': None, 'logits': logits}

    # Initialize and Load YOUR Fine-Tuned Weights
    device = torch.device('cpu') # Streamlit Community Cloud uses CPU
    model = CaduceusClassifier(config=config, num_labels=5).to(device)
    
    # Path to your uploaded weights
    weights_path = 'model_epoch_1.pth' 
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        st.error("model_epoch_1.pth not found in repository!")
        
    model.eval()
    return model, tokenizer, config

# Load everything
with st.spinner("Loading Caduceus Model & Weights... (This takes ~1 minute on first run)"):
    model, tokenizer, config = load_model_and_tokenizer()

st.success("✅ Model loaded successfully!")

# ==========================================
# 2. UI LAYOUT
# ==========================================
st.set_page_config(page_title="Rare Disease DNA Analyzer", layout="wide")
st.title("🧬 Rare Disease DNA Analyzer")
st.markdown("Paste a DNA sequence below to predict its clinical significance using a fine-tuned Caduceus (Mamba) model.")

# Input
dna_input = st.text_area("Enter DNA Sequence (A, T, C, G):", height=150, placeholder="e.g., ATGCATGC...")

# ==========================================
# 3. INFERENCE & VISUALIZATION
# ==========================================
if st.button("🔍 Analyze Sequence", type="primary"):
    if not dna_input:
        st.warning("Please enter a DNA sequence.")
    else:
        # Clean input
        seq = dna_input.upper().replace(" ", "").replace("\n", "")
        
        if not all(c in "ATCG" for c in seq):
            st.error("Invalid characters! Please use only A, T, C, G.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Prediction Results")
                # Tokenize
                inputs = tokenizer(seq, return_tensors='pt', padding='max_length', max_length=1024, truncation=True)
                
                # Predict
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs['logits'], dim=1)[0]
                
                labels = ['Benign', 'Likely Benign', 'VUS', 'Likely Pathogenic', 'Pathogenic']
                
                # Display results
                for label, prob in zip(labels, probs):
                    st.metric(label=f"{label}", value=f"{prob.item()*100:.2f}%")
                
                # Highlight top prediction
                top_idx = torch.argmax(probs).item()
                st.success(f"**Top Prediction:** {labels[top_idx]} with {probs[top_idx].item()*100:.2f}% confidence.")

            with col2:
                st.subheader("🧬 3D Structure Visualization")
                # Generate 3D view of the first 100 bases
                view_seq = seq[:100] 
                
                view = py3Dmol.view(width=400, height=400)
                view.setBackgroundColor('0x1a1a1a')
                
                radius = 10.0
                rise_per_bp = 3.4
                twist_angle = 36.0
                
                for i in range(len(view_seq)):
                    angle = math.radians(i * twist_angle)
                    z = i * rise_per_bp
                    x1, y1 = radius * math.cos(angle), radius * math.sin(angle)
                    x2, y2 = radius * math.cos(angle + math.pi), radius * math.sin(angle + math.pi)
                    
                    if i > 0:
                        prev_angle = math.radians((i-1) * twist_angle)
                        prev_z = (i-1) * rise_per_bp
                        px1, py1 = radius * math.cos(prev_angle), radius * math.sin(prev_angle)
                        px2, py2 = radius * math.cos(prev_angle + math.pi), radius * math.sin(prev_angle + math.pi)
                        view.addCylinder({'start': {'x': px1, 'y': py1, 'z': prev_z}, 'end': {'x': x1, 'y': y1, 'z': z}, 'radius': 0.8, 'color': 'cyan'})
                        view.addCylinder({'start': {'x': px2, 'y': py2, 'z': prev_z}, 'end': {'x': x2, 'y': y2, 'z': z}, 'radius': 0.8, 'color': 'magenta'})
                        
                    view.addCylinder({'start': {'x': x1, 'y': y1, 'z': z}, 'end': {'x': x2, 'y': y2, 'z': z}, 'radius': 0.5, 'color': 'yellow'})
                    
                    c1 = {'A': 'red', 'T': 'green', 'C': 'blue', 'G': 'orange'}.get(view_seq[i], 'white')
                    c2 = {'A': 'red', 'T': 'green', 'C': 'blue', 'G': 'orange'}.get({'A':'T','T':'A','C':'G','G':'C'}.get(view_seq[i], 'N'), 'white')
                    view.addSphere({'center': {'x': x1, 'y': y1, 'z': z}, 'radius': 1.2, 'color': c1})
                    view.addSphere({'center': {'x': x2, 'y': y2, 'z': z}, 'radius': 1.2, 'color': c2})

                view.zoomTo()
                components.html(view._make_html(), height=450)
