# Grammatical Metaphor Detection using Deep Learning

## Project Overview

This project develops a sophisticated deep learning model to automatically detect and classify grammatical metaphors in natural language text. Grammatical metaphors are figures of speech where grammatical structures are used metaphorically, representing a complex linguistic phenomenon that requires advanced NLP techniques to identify accurately.

The model achieves **95.27% test accuracy** using a DistilBERT-based architecture with PyTorch, demonstrating the effectiveness of transformer models for linguistic analysis tasks.

## Problem Statement

Grammatical metaphors are subtle linguistic phenomena that play a crucial role in language comprehension and analysis. Manual detection is time-consuming and prone to human error. This project automates the detection process using state-of-the-art deep learning techniques, enabling rapid and accurate identification of grammatical metaphors across large text corpora.

## Dataset

- **Total Samples**: 17,967 annotated examples
- **Training Set**: 14,374 samples (80%)
- **Validation Set**: 1,796 samples (10%)
- **Test Set**: 1,797 samples (10%)
- **Annotation**: High-quality manual annotations with expert linguistic review

## Technologies & Libraries

### Core Technologies
- **Deep Learning Framework**: PyTorch
- **Pre-trained Model**: DistilBERT (Hugging Face Transformers)
- **Tokenization**: BertTokenizer
- **Data Processing**: Pandas, NumPy

### Supporting Libraries
- **Evaluation Metrics**: Scikit-learn (accuracy_score, f1_score, classification_report, confusion_matrix)
- **Data Loading**: PyTorch Dataset and DataLoader
- **Visualization**: Matplotlib, Seaborn
- **Warnings Management**: Python warnings module

## Model Architecture

### Approach
The project employs a **dual-representation strategy** with fine-tuned transformer embeddings:

1. **Tokenization Phase**
   - Text input tokenized using BertTokenizer
   - Sequence length standardized for consistent processing
   - Special tokens ([CLS], [SEP]) added appropriately

2. **Embedding & Encoding**
   - DistilBERT processes tokenized input
   - Contextual embeddings capture linguistic nuances
   - Attention mechanisms learn metaphor patterns

3. **Classification Layer**
   - Dense layers process DistilBERT output
   - Softmax activation for binary/multi-class classification
   - Dropout for regularization to prevent overfitting

4. **Attention Masking**
   - Padding tokens masked during attention computation
   - Ensures model focuses on meaningful content
   - Improves robustness and generalization

## Results & Performance Metrics

### Overall Performance
- **Test Accuracy**: 95.27%
- **F1 Score**: 0.9526
- **Robustness (without target hints)**: 93%

### Detailed Analysis
- **Occurrence-level Breakdown**: 100% accuracy on grammatical metaphor occurrences 3-4
- **Error Analysis**: Model primarily struggles with edge cases and rare linguistic constructs
- **Consistency**: High performance across different text domains and sentence structures

### Confusion Matrix & Classification Report
- Detailed breakdown available in project notebooks
- Precision and recall metrics indicate balanced performance
- Low false positive rate ensures practical applicability

## Installation & Setup

### Requirements
```
torch>=2.0.0
pandas>=1.5.3
numpy>=1.24.3
transformers>=4.30.0
scikit-learn>=1.2.2
matplotlib>=3.7.1
seaborn>=0.12.2
jupyter>=1.0.0
```

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/vicky805515/grammatical-metaphor-detection.git
cd grammatical-metaphor-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Model

1. **Load and preprocess data**:
```python
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
# Load your data and tokenize
```

2. **Train the model**:
```bash
jupyter notebook grammatical_metaphor_detection.ipynb
# Follow cells in sequence to preprocess, train, and evaluate
```

3. **Make predictions on new text**:
```python
text = "Your sample sentence here"
# Model will classify as grammatical metaphor or not
```

## Project Workflow

[View flowchart diagram in repository]

The project follows a structured pipeline:
1. **Data Exploration** - Analyze dataset characteristics and distribution
2. **Preprocessing** - Tokenization, sequence padding, attention mask creation
3. **Model Development** - Fine-tune DistilBERT for the classification task
4. **Training** - Iterative training with validation monitoring
5. **Evaluation** - Comprehensive metrics and error analysis
6. **Inference** - Deploy model for real-world predictions

## Key Findings & Insights

### What Works Well
- Transformer-based models excel at capturing contextual linguistic patterns
- DistilBERT provides excellent performance with reduced computational requirements
- Attention mechanisms effectively identify metaphor-bearing spans in text

### Model Limitations
- Performance varies on rare or novel linguistic constructs
- Edge cases with ambiguous grammatical structures pose challenges
- Domain-specific metaphors may require additional training data

### Robustness
- 93% accuracy maintained when removing direct target hints
- Model learns genuine linguistic patterns, not surface-level heuristics
- Generalizes well across different text styles and domains

## Error Analysis

The model's misclassifications primarily occur in:
- **Ambiguous cases**: Sentences with borderline metaphor usage
- **Rare constructs**: Unusual grammatical structures with limited training examples
- **Domain-specific language**: Technical or specialized terminology requiring domain knowledge

Detailed error analysis available in the Jupyter notebook with example misclassifications and explanations.

## Future Improvements

### Short-term Enhancements
- Ensemble methods combining multiple transformer architectures
- Incorporation of linguistic features (POS tags, dependency parsing)
- Domain-specific fine-tuning datasets

### Long-term Research Directions
- Multi-lingual grammatical metaphor detection
- Fine-grained metaphor type classification
- Integration with downstream NLP tasks
- Web application for real-time metaphor detection
- Interpretability analysis using SHAP or attention visualization

## Files in Repository

- `grammatical_metaphor_detection.ipynb` - Complete Jupyter notebook with full implementation
- `requirements.txt` - Python package dependencies
- `data/` - Dataset folder (training, validation, test splits)
- `flowchart.png` - Project workflow visualization
- `README.md` - Project documentation (this file)

## Key Learnings

1. **Transformer Models for NLP**: Pre-trained models like DistilBERT dramatically improve performance on specialized linguistic tasks
2. **Data Quality Matters**: High-quality annotations and balanced datasets are crucial for model reliability
3. **Contextual Understanding**: Attention mechanisms enable models to understand metaphorical language in context
4. **Iterative Refinement**: Model performance improves significantly through hyperparameter tuning and architecture modifications

## How to Cite This Project

If you use this project in your research or work, please cite:

```
Vignesh Arvind. "Grammatical Metaphor Detection using Deep Learning." 
Master's thesis, Macquarie University, 2025.
```

## Contact & Collaboration

- **Email**: vignesh.arvind@students.mq.edu.au
- **LinkedIn**: [linkedin.com/in/vignesh-arvind](https://linkedin.com/in/vignesh-arvind)
- **GitHub**: [@vicky805515](https://github.com/vicky805515)

Feel free to reach out for discussions, collaborations, or questions about this project!

## License

This project is open source and available under the MIT License.

---

**Last Updated**: December 2025  
**Status**: Complete - Thesis Project
