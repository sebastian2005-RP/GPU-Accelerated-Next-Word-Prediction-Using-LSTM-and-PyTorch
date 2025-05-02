# 🔮 Next-Word Prediction Using LSTM with PyTorch (GPU-Accelerated)

This repository presents a **next-word prediction model** built with **PyTorch** using an **LSTM architecture**, optimized for **GPU acceleration**. It includes everything from text preprocessing to model training and next-word generation. This serves as an educational and practical introduction to natural language generation using deep learning.

---

Features

- ✅ Tokenization and preprocessing using NLTK
- ✅ Dynamic vocabulary creation with `<UNK>` token handling
- ✅ Training data preparation with padded sequences
- ✅ Custom PyTorch Dataset and DataLoader
- ✅ LSTM-based neural network with embedding and linear layers
- ✅ GPU support via CUDA (if available)
- ✅ Next-word prediction from user-defined input
- ✅ Loop-based text generation for full sentence construction

---

Example Output

```python
Input: "Zero-shot learning"
Generated: "Zero-shot learning is a powerful AI technique that enables models to learn new"
Install dependencies:

bash
Copy
Edit
pip install torch nltk
Download required NLTK data:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Usage
Set your training data
Modify the document string in the script with your own dataset (e.g., "Exploring Data Availability in LLM Development").

Train the model
Run the full script to tokenize, create sequences, and train the LSTM model.

Generate predictions
Use the prediction(model, vocab, text) function to predict the next word.

Generate full sequences
The script includes a loop to auto-generate multiple words:

python
Copy
Edit
input_text = "Zero-shot learning"
for _ in range(25):
    output = prediction(model, vocab, input_text)
    print(output)
    input_text = output
Model Architecture
Embedding Layer: Converts token IDs to dense vectors

LSTM Layer: Learns temporal dependencies

Linear Layer: Outputs scores for each vocabulary word

Requirements
Python 3.x

PyTorch

NLTK

Applications
Language modeling

Predictive text systems

Educational demonstrations in NLP and deep learning

Building blocks for advanced generative AI models
