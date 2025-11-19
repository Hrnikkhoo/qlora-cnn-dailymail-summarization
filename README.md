# QLoRA Fine-tuning for News Summarization

This project implements fine-tuning a generative language model (TinyLlama) using **QLoRA (Quantized LoRA)** on the **CNN/DailyMail** dataset to generate news article summaries (highlights).

## 📋 Project Overview

The project includes:
- **Model Setup**: Loading and quantizing a language model (4-bit quantization)
- **Data Preparation**: Processing CNN/DailyMail dataset for summarization task
- **QLoRA Configuration**: Setting up Parameter-Efficient Fine-Tuning with LoRA
- **Training**: Fine-tuning the model to generate summaries from news articles
- **Evaluation**: Assessing model performance using ROUGE metrics and human evaluation
- **Analysis**: Final performance analysis and insights

## 🎯 Objectives

- Fine-tune a generative language model using PEFT (QLoRA) method
- Train the model to generate summaries (highlights) from news articles
- Evaluate model performance using ROUGE scores
- Analyze the model's ability to identify important parts of news articles

## 🛠️ Technologies Used

- **Transformers**: Hugging Face Transformers library
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA)
- **BitsAndBytes**: 4-bit quantization for memory efficiency
- **Datasets**: Hugging Face Datasets for data loading
- **ROUGE Score**: For evaluation metrics
- **PyTorch**: Deep learning framework

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- At least 8GB GPU memory (with 4-bit quantization)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qlora-news-summarization.git
cd qlora-news-summarization
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Notebook

1. Open `main1.ipynb` in Jupyter Notebook or JupyterLab
2. Execute cells sequentially:
   - **Cell 1**: Install required libraries
   - **Cell 2**: Import libraries
   - **Cell 3**: Load and quantize the model (TinyLlama)
   - **Cell 4**: Load CNN/DailyMail dataset
   - **Cell 5**: Preprocess data
   - **Cell 6**: Configure QLoRA
   - **Cell 7**: Train the model
   - **Cell 8**: Save model and generate summaries
   - **Cell 9**: Evaluate with ROUGE metrics
   - **Cell 10**: Human evaluation

### Model Selection

You can change the model in Cell 3. Supported models:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default - lightweight and fast)
- `google/gemma-2b-it`
- `microsoft/phi-2`
- `mistralai/Mistral-7B-Instruct-v0.2`

### Training Configuration

Key training parameters (adjustable in Cell 7):
- **Epochs**: 3
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4
- **Learning Rate**: 2e-4
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32

## 📊 Dataset

- **Dataset**: CNN/DailyMail v3.0.0
- **Task**: Abstractive Summarization
- **Input**: News articles
- **Output**: Highlights (summaries)
- **Training Samples**: 5,000 (configurable)
- **Validation Samples**: 500 (configurable)

## 🔧 Model Architecture

- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Quantization**: 4-bit NF4 quantization
- **Fine-tuning Method**: QLoRA (Quantized LoRA)
- **Trainable Parameters**: Only LoRA adapters (~1% of total parameters)

## 📈 Evaluation Metrics

The project uses **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metrics:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

## 📁 Project Structure

```
qlora-news-summarization/
├── main1.ipynb              # Main notebook with all code
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore file
├── results/                 # Training outputs (generated)
├── fine_tuned_model/        # Saved fine-tuned model (generated)
└── venv/                    # Virtual environment (not in git)
```

## 💾 Model Storage

### Base Model
- **Location**: `~/.cache/huggingface/hub/`
- **Size**: ~2.2 GB (downloaded once, cached for future use)
- **Source**: Hugging Face Hub

### Fine-tuned Model (LoRA Adapter)
- **Location**: `./fine_tuned_model/`
- **Size**: Small (few MB to tens of MB)
- **Contains**: LoRA adapter weights, tokenizer, configuration files

## 📝 Key Features

✅ **Memory Efficient**: 4-bit quantization reduces RAM/GPU usage significantly  
✅ **Parameter Efficient**: Only trains LoRA adapters (~1% of parameters)  
✅ **Fast Training**: QLoRA enables training on limited hardware  
✅ **Comprehensive Evaluation**: ROUGE metrics + human evaluation  
✅ **Easy to Use**: Well-documented notebook with clear sections  

## 🎓 Learning Outcomes

This project demonstrates:
1. How to use QLoRA for efficient fine-tuning
2. Working with large language models on limited hardware
3. Data preprocessing for summarization tasks
4. Model evaluation using ROUGE metrics
5. Best practices for fine-tuning language models

## ⚠️ Limitations & Improvements

### Current Limitations
- Uses subset of dataset (5,000 training samples) for speed
- Small model (TinyLlama 1.1B) may miss some important details
- Limited to 3 epochs

### Potential Improvements
- Increase number of training epochs
- Increase LoRA rank (r) for more trainable parameters
- Use larger base model
- Train on full dataset
- Experiment with different prompt formats
- Add more evaluation metrics (BLEU, METEOR, etc.)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Hugging Face for Transformers and Datasets libraries
- PEFT library developers
- CNN/DailyMail dataset creators
- TinyLlama model developers

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is for educational purposes. For production use, consider:
1. Training on full dataset
2. More hyperparameter tuning
3. Comprehensive evaluation with multiple metrics
4. Testing on different datasets

