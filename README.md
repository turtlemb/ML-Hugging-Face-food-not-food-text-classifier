# 🍔 Food or Not Food Text Classifier with Hugging Face Transformers

This project builds a **binary text classifier** that predicts whether a sentence is about food or not, using **transfer learning** and **Hugging Face Transformers**. It leverages a synthetic dataset generated with Mistral Chat/Mixtral and deploys the model as a live demo using Gradio and Hugging Face Spaces.

---

## 🧠 Project Overview

The task is to classify short text captions into two categories:
- **Food**
- **Not Food**

The project fine-tunes a pre-trained **DistilBERT-base-uncased** model, a lighter and faster version of BERT suitable for classification tasks. This model is trained on a curated set of synthetic food-related and non-food captions, and exposed to increasingly ambiguous examples to improve robustness.

---

## 📂 Dataset

- **Source**: Synthetic (generated via Mistral Chat / Mixtral)
- **Size**: 270 labeled captions
- **Classes**: Binary (`food`, `not food`)
- **Data type**: Structured text data

---

## 🎯 Problem Statement

Develop a binary classifier that:
- Accurately predicts whether a sentence is related to food
- Achieves high **accuracy** on both clear and ambiguous text inputs

---

## 🛠️ Tools & Technologies

- **Python** (via Google Colab)
- **Datasets** (via Google Colab)
- **Evaluate** (via Google Colab)
- **Accelerate** (via Google Colab)
- **Transformers** 4.48.3  
- **PyTorch** 2.6.0    
- **Gradio** 5.20.0  
- **Hugging Face Hub**

---

## 🔍 Notebook Structure

### 1. Setup & Preprocessing
- Import libraries  
- Inspect random samples  
- Prepare data and labels  
- Split into training and test sets  
- Tokenize text data  
- Define preprocessing functions  

### 2. Model Training & Evaluation
- Set up evaluation metric  
- Load pre-trained DistilBERT model  
- Count model parameters  
- Configure Hugging Face `Trainer`  
- Train the model  
- Save the model  
- Visualize training metrics (loss curves)  

### 3. Prediction & Inference
- Make predictions on the test set  
- Evaluate predictions and probabilities  
- Test model on custom text inputs  
- Use pipeline and batch modes  
- Time model predictions across input sizes  
- Run inference manually via PyTorch  

### 4. Deployment (Gradio + Hugging Face Spaces)
- Turn model into an inference function  
- Build a local Gradio demo  
- Create required files (`app.py`, `README.md`, `requirements.txt`)  
- Push demo to Hugging Face Spaces  
- Embed and share the deployed app

---

## 📈 Results

- The model achieves high accuracy on clear examples  
- Performance improves with data augmentation and diverse examples  
- **Live Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/turtlemb/hf_extended_food_not_food_text_classifier)

---

## 🚀 Future Improvements

- Expand dataset with more ambiguous or nuanced food-related sentences  
- Include shorter, informal, or emoji-containing text  
- Experiment with other transformer architectures (e.g., RoBERTa, DeBERTa)  
- Fine-tune hyperparameters and regularization techniques

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Hugging Face for models and deployment platform  
- Mistral Chat / Mixtral for synthetic data generation  
- Gradio for easy web demo creation  
