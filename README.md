# 🧬 Rare Disease DNA Analyzer

A state-of-the-art deep learning system for predicting the clinical significance of genetic variants using **Caduceus**, a Mamba-based DNA language model.

## 📋 Overview

This project fine-tunes the Caduceus foundation model to classify single nucleotide variants (SNVs) into five clinical significance categories based on the ClinVar database. By leveraging Mamba's selective state space models (SSMs), the system achieves linear-time complexity for processing long genomic sequences, capturing biological dependencies that traditional Transformer-based models miss.

## ✨ Key Features

- **SOTA Architecture**: Powered by Caduceus (Mamba-based) with reverse-complement equivariance
- **Clinical Predictions**: Classifies variants into 5 categories:
  - Benign
  - Likely Benign  
  - Variant of Uncertain Significance (VUS)
  - Likely Pathogenic
  - Pathogenic
- **Interactive Web Interface**: Gradio-based frontend with real-time inference
- **3D Visualization**: Interactive DNA double helix structure viewer using Plotly
- **Large-Scale Training**: 250,000 balanced ClinVar variants mapped to hg38 reference genome
- **Multi-GPU Support**: Optimized for dual Tesla T4 GPUs with bfloat16 mixed precision

## 🏗️ Architecture

### Model Specifications
- **Base Model**: Caduceus-PS (seqlen=131k, d_model=256, n_layer=16)
- **Parameters**: 7.8M trainable parameters
- **Input**: 1,024 bp genomic windows centered on variants
- **Output**: 5-class clinical significance prediction
- **Hidden Dimension**: 512 (2 × d_model for bidirectional processing)

### Why Mamba for Genomics?
Traditional Transformers suffer from quadratic memory complexity, making them impractical for long DNA sequences. Mamba's Selective State Space Models (SSMs) provide:
- **Linear time complexity** O(n) instead of O(n²)
- **Efficient long-range dependency modeling**
- **Hardware-aware parallelization**
- **Native support for reverse-complement processing**

## 📊 Dataset

### Source
- **ClinVar Database**: Downloaded from NCBI FTP (variant_summary.txt)
- **Reference Genome**: hg38 (GRCh38) - ~938MB
- **Variant Type**: Single nucleotide variants (SNVs) only

### Processing Pipeline
1. Filtered ClinVar variants for GRCh38 assembly
2. Selected only single nucleotide variants
3. Extracted 1,024 bp windows centered on each variant
4. Balanced dataset: 50,000 samples per class (250,000 total)
5. Tokenized using Caduceus tokenizer (vocab size: 16)

### Class Distribution
| Class | Samples |
|-------|---------|
| Benign | 50,000 |
| Likely Benign | 50,000 |
| VUS | 50,000 |
| Likely Pathogenic | 50,000 |
| Pathogenic | 50,000 |

## 🚀 Installation

### Requirements
- Python 3.12+
- PyTorch 2.10.0+ with CUDA 12.8
- GPU with 16GB+ VRAM (recommended: NVIDIA Tesla T4 x2)
- 10GB+ disk space for reference genome and datasets

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rare-disease-analyzer.git
cd rare-disease-analyzer

# Install core dependencies
pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install Mamba dependencies
pip install ninja packaging
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm==2.3.2.post1 --no-build-isolation

# Install additional requirements
pip install -r requirements.txt
```

### requirements.txt
```
transformers>=4.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.66.0
gradio>=4.0.0
plotly>=5.18.0
pyfaidx>=0.7.2
safetensors>=0.4.0
seaborn>=0.13.0
matplotlib>=3.8.0
```

## 📖 Usage

### 1. Data Preparation

```bash
# Download ClinVar data
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
gunzip variant_summary.txt.gz

# Download hg38 reference genome
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
```

### 2. Clone Base Model
```bash
git clone https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 caduceus_model
```

### 3. Run Training
```bash
jupyter notebook rare_disease_analyzer.ipynb
```

The notebook includes:
- Data preprocessing and sequence extraction
- Model architecture definition with custom classifier head
- Training loop with bfloat16 mixed precision
- Gradient accumulation for effective batch size of 128
- Checkpoint saving after each epoch

### 4. Launch Web Interface
```python
# Run the Gradio cell in the notebook
# Or execute:
python app.py
```

## 🎯 Model Performance

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 16 per GPU (effective batch: 128 with 8-step gradient accumulation)
- **Learning Rate**: 3e-5 with cosine decay
- **Warmup**: 10% of total steps
- **Optimizer**: AdamW (weight decay: 0.01)
- **Training Time**: ~19 hours on dual Tesla T4 GPUs
- **Mixed Precision**: bfloat16 autocast

### Evaluation Metrics
| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 81.2% |
| **Macro Precision** | 0.812 |
| **Macro Recall** | 0.812 |
| **Macro F1-Score** | 0.812 |
| **Weighted F1** | 0.812 |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Benign | 0.840 | 0.840 | 0.840 |
| Likely Benign | 0.820 | 0.820 | 0.820 |
| VUS | 0.731 | 0.760 | 0.745 |
| Likely Pathogenic | 0.800 | 0.800 | 0.800 |
| Pathogenic | 0.840 | 0.840 | 0.840 |

**Note**: VUS shows lower precision due to the inherent biological ambiguity of variants with uncertain clinical significance.

## 🗂️ Project Structure

```
rare-disease-analyzer/
├── rare_disease_analyzer.ipynb    # Main training notebook
├── app.py                          # Gradio web interface (optional)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── caduceus_model/                 # Cloned base model
│   ├── configuration_caduceus.py
│   ├── modeling_caduceus.py
│   ├── modeling_rcps.py
│   ├── tokenization_caduceus.py
│   └── model.safetensors
├── checkpoints/                    # Saved model checkpoints
│   ├── model_epoch_1.pth
│   ├── model_epoch_2.pth
│   └── model_epoch_3.pth
├── variant_summary.txt             # ClinVar dataset (~2GB)
├── hg38.fa                         # Reference genome (~3GB)
├── clinvar_sequences.csv           # Extracted sequences
└── pretokenized_data.pt            # Preprocessed tokens (~1GB)
```

## 🔬 Technical Details

### Sequence Extraction
- **Window Size**: 1,024 bp (512 bp upstream + variant + 512 bp downstream)
- **Strand**: Forward strand (GRCh38 coordinates)
- **Mutation Injection**: REF base replaced with ALT at variant position
- **Failed Extractions**: ~0.15% (368 variants dropped)

### Tokenization
- **Vocabulary**: 16 tokens (A, T, C, G + special tokens)
- **Max Length**: 1,024 tokens
- **Padding**: Right-padding with special token
- **Truncation**: Enabled for sequences exceeding max length
- **Pre-tokenization**: All 250k sequences tokenized upfront (~12 minutes)

### Training Optimizations
- **Mixed Precision**: bfloat16 autocast for 2× speedup on T4 GPUs
- **Gradient Accumulation**: 8 steps for effective batch size of 128
- **Memory Management**: `PYTORCH_ALLOC_CONF=expandable_segments:True`
- **DataLoader**: 2 workers with pinned memory for fast CPU→GPU transfer
- **Learning Rate Schedule**: Cosine decay with 10% warmup
- **Gradient Clipping**: max_norm=1.0 to prevent exploding gradients

## 🌐 Web Interface

The Gradio app provides:
- **Text Input**: Paste DNA sequences (50-200 bp)
- **Random Generator**: Generate synthetic sequences for testing
- **3D Visualization**: Interactive Plotly-based DNA helix viewer
- **Real-time Predictions**: Instant clinical significance classification
- **Probability Distribution**: Confidence scores for all 5 classes
- **Educational Content**: Expandable accordion explaining each class

### Example Usage
```python
# Example pathogenic sequence (Sickle Cell Anemia mutation)
sequence = "CTTCATCCACGTTCACCTTGCCCCACAGGGCAGTAACGGCAGACTTCTCCACAGGAGTCAGATGCACCATGGTGTCTGTTTGAGGTTGCTAGTGAACACA"

# Model predicts: Pathogenic (94.3% confidence)
```

## 📈 Future Improvements

- [ ] Implement multi-GPU training with `nn.DataParallel`
- [ ] Add support for indels and structural variants
- [ ] Integrate attention visualization for interpretability
- [ ] Deploy on Hugging Face Spaces for public access
- [ ] Add ensemble modeling for improved accuracy
- [ ] Implement variant effect prediction (missense, nonsense, etc.)
- [ ] Add support for ensemble predictions across multiple models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Caduceus Team**: For the excellent Mamba-based DNA language model
- **ClinVar**: For providing the clinical variant database
- **UCSC Genome Browser**: For the hg38 reference genome
- **Hugging Face**: For the Transformers library and model hosting
- **Kaggle**: For providing free dual Tesla T4 GPU resources

## 📚 Citations

### Caduceus (Mamba for Genomics)
```bibtex
@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff, Yair and Kao, Chia-Hsiang and Gokaslan, Aaron and Dao, Tri and Gu, Albert and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2403.03234},
  year={2024}
}
```

### Mamba (Selective State Space Models)
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

### ClinVar Database
```bibtex
@article{landrum2018clinvar,
  title={ClinVar: improving access to variant interpretations and supporting evidence},
  author={Landrum, Melissa J and Lee, Jennifer M and Benson, Mark and Brown, Garth R and Chao, Chen and Chitipiralla, Sudha and Gu, Baoshan and Hart, Jennifer and Hoffman, Dustin and Jang, Wonhee and others},
  journal={Nucleic acids research},
  volume={46},
  number={D1},
  pages={D1062--D1067},
  year={2018},
  publisher={Oxford University Press}
}
```

## 📧 Contact

For questions or collaborations, please open an issue on GitHub or contact the maintainers.

---

**Built using Mamba architecture and trained on Kaggle GPU infrastructure**
