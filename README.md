# GPU-Accelerated Next Word Prediction Using LSTM and PyTorch 🚀

![Next Word Prediction](https://img.shields.io/badge/Next%20Word%20Prediction-GPU%20Accelerated-blue)

Welcome to the **GPU-Accelerated Next Word Prediction Using LSTM and PyTorch** repository! This project showcases an efficient model for predicting the next word in a sequence using Long Short-Term Memory (LSTM) networks. By leveraging the power of GPUs and the PyTorch framework, this implementation aims to enhance the speed and accuracy of next-word predictions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Generating Predictions](#generating-predictions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Releases](#releases)

## Introduction

In the age of digital communication, next-word prediction plays a crucial role in enhancing user experience. This repository implements a next-word prediction model using LSTM networks, optimized for GPU processing. The model takes a sequence of words as input and predicts the most likely next word based on the context provided.

## Features

- **GPU Acceleration**: Utilizes GPU for faster training and prediction.
- **LSTM Architecture**: Implements Long Short-Term Memory networks for effective sequence modeling.
- **Data Preprocessing**: Includes tokenization and vocabulary creation using NLTK.
- **Text Generation**: Generates coherent text predictions based on input phrases.
- **Easy to Use**: Simple API for generating predictions.

## Getting Started

### Prerequisites

To run this project, you need the following software installed on your machine:

- Python 3.6 or higher
- PyTorch (with GPU support)
- NLTK
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sebastian2005-RP/GPU-Accelerated-Next-Word-Prediction-Using-LSTM-and-PyTorch.git
   cd GPU-Accelerated-Next-Word-Prediction-Using-LSTM-and-PyTorch
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

Before training the model, you need to prepare your dataset. This involves tokenizing your text and creating a vocabulary. You can use the provided scripts in the `data_preprocessing` folder.

1. Place your text data in the `data` folder.
2. Run the data preprocessing script:

   ```bash
   python data_preprocessing/preprocess.py
   ```

This will generate the necessary tokenized data and vocabulary files.

## Model Architecture

The model architecture consists of an LSTM layer followed by a fully connected layer. This setup allows the model to learn dependencies over long sequences of text. 

### LSTM Layer

- **Input Size**: The size of the vocabulary.
- **Hidden Size**: Number of features in the hidden state.
- **Number of Layers**: Stacked LSTM layers for better learning.

### Fully Connected Layer

- Takes the output from the LSTM layer and predicts the next word based on the learned features.

## Training the Model

To train the model, use the following command:

```bash
python train.py --epochs 10 --batch_size 64
```

Adjust the `epochs` and `batch_size` parameters as needed. The model will save the trained weights to the `models` directory.

## Generating Predictions

Once the model is trained, you can generate predictions using the provided script:

```bash
python predict.py --input "Your input phrase here"
```

This will output the predicted next word based on the input phrase.

## Usage

To utilize the model in your applications, follow these steps:

1. Load the trained model.
2. Preprocess your input text.
3. Call the prediction function.

### Example

```python
import torch
from model import LSTMModel

# Load the model
model = LSTMModel()
model.load_state_dict(torch.load('models/model_weights.pth'))

# Prepare input
input_text = "The quick brown fox"
# Generate prediction
predicted_word = model.predict(input_text)
print(predicted_word)
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, you can reach out to the maintainer:

- **Name**: Sebastian
- **Email**: sebastian@example.com

## Releases

To download the latest release of this project, visit the [Releases section](https://github.com/sebastian2005-RP/GPU-Accelerated-Next-Word-Prediction-Using-LSTM-and-PyTorch/releases). 

You can also check for updates and new features regularly.

---

Thank you for checking out this project! We hope it helps you in your journey of exploring next-word prediction using deep learning techniques. Happy coding!