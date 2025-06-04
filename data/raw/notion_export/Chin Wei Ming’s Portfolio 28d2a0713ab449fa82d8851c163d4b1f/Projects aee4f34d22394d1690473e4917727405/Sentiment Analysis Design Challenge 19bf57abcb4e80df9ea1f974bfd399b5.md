# Sentiment Analysis Design Challenge

Timeline: September 9, 2024 → December 20, 2024
Client: 50.040: Natural Language Processing, Prof. Lu Wei
My Role: Machine Learning Engineer, NLP Engineer
Deliverables: Exploration of NLP model architectures., Refinement for improved model performance
Tools: Gensim, Pandas, Python, Pytorch, Transformer (Hugging Face), Vast.ai
Document: ../NLP_final_project_report.pdf

![Screenshot 2025-02-15 at 12.19.25 PM.png](Sentiment%20Analysis%20Design%20Challenge%2019bf57abcb4e80df9ea1f974bfd399b5/Screenshot_2025-02-15_at_12.19.25_PM.png)

## Project Overview

---

Sentiment analysis is a crucial Natural Language Processing (NLP) task used in various industries, from customer feedback analysis to financial market predictions. This project explores different deep learning architectures to enhance sentiment classification performance on the **IMDB movie reviews dataset**, a widely used benchmark for binary sentiment analysis (positive vs. negative reviews). Our final model, **LoRA-RoBERTa**, achieved the **highest performance**, with an **F1-score of 0.9407** on the test set, demonstrating its ability to accurately capture nuanced sentiment in text. This project highlights the power of **transformer-based models** and **efficient fine-tuning techniques** in advancing sentiment analysis.

## Approach & Methodology

---

To systematically improve sentiment classification performance, we experimented with various deep learning architectures, progressively enhancing model accuracy and interpretability.

### **Baseline Models: BiRNN & TextCNN**

We started with **BiRNN (Bidirectional Recurrent Neural Network)** and **TextCNN (Text-based Convolutional Neural Network)** as our baseline models.

- **BiRNN** captured sequential dependencies in text but had limitations in handling long-range dependencies.
- **TextCNN**, leveraging convolutional layers, excelled at recognizing local n-gram patterns but lacked sequential understanding.
- Performance Benchmark: **BiRNN achieved 86.48% accuracy, while TextCNN underperformed at 54.94% accuracy**, highlighting the need for further improvements.

### **Ensemble Model: Attention-Based Aggregation**

To leverage the strengths of both BiRNN and TextCNN, we implemented an **ensemble model** using an **attention-based weighted aggregation mechanism**:

- **Simple Ensemble:** Averaged predictions from BiRNN and TextCNN.
- **Attention-Based Ensemble:** Assigned dynamic weights to each model’s output based on contextual importance.
- **Custom Embeddings:** Replaced standard GloVe embeddings with domain-specific Word2Vec embeddings for richer feature representation.
- **Results:** This approach improved performance, achieving **92.67% accuracy and an F1-score of 0.9273**, but we aimed for further optimization.

### **RoBERTa-BiLSTM: Contextual Embeddings + Sequential Understanding**

To incorporate **pre-trained language models**, we introduced **RoBERTa** to generate deep contextual embeddings and **BiLSTM (Bidirectional Long Short-Term Memory)** to capture sequential dependencies.

- **RoBERTa provided robust sentence-level representations**, improving sentiment understanding.
- **BiLSTM processed embeddings bidirectionally**, retaining contextual dependencies across longer text spans.
- **Results:** This model further improved accuracy to **93.92% with an F1-score of 0.9393**.

### **LoRA-RoBERTa: Efficient Transformer Fine-Tuning (Final Model)**

Given computational constraints, we fine-tuned **RoBERTa-large** using **Low-Rank Adaptation (LoRA)**, a parameter-efficient tuning method.

- **Why LoRA?** Instead of updating all 355M parameters in RoBERTa-large, LoRA fine-tuned only a subset of trainable parameters, significantly reducing memory requirements while maintaining high performance.
- **Optimization Techniques:** Used **AdamW optimizer, gradient clipping, and linear learning rate scheduling** for stable training.
- **Final Results:** **LoRA-RoBERTa achieved the best performance with an F1-score of 0.9407**, outperforming all previous models.

## Results

---

Through systematic experimentation, we observed consistent improvements in sentiment classification performance as we transitioned from traditional deep learning models to **transformer-based architectures**.

| Model | Accuracy | Precision | Recall | F1 score |
| --- | --- | --- | --- | --- |
| **BiRNN (Baseline)** | 86.48% | 83.85% | 90.34% | 86.97% |
| **TextCNN (Baseline)** | 54.94% | 52.61% | 98.88% | 68.67% |
| **Ensemble Model** | 92.67% | 91.90% | 93.58% | 92.73% |
| **RoBERTa-BiLSTM** | 93.92% | 93.68% | 94.18% | 93.93% |
| **LoRA-RoBERTa** | 94.08% | 94.25% | 93.88% | 94.07% |

📌 **Key Takeaways:**

- **BiRNN performed well but struggled with long-range dependencies.**
- **TextCNN underperformed due to its lack of sequential context.**
- **The Ensemble model improved performance** by combining both architectures with attention-based weighting.
- **RoBERTa-BiLSTM outperformed the ensemble model** by leveraging contextual embeddings with sequential processing.
- **LoRA-RoBERTa achieved the best results, demonstrating the effectiveness of transformer-based fine-tuning.**

## Challenges

---

🔹 **Computational Constraints:**

- **Challenge:** Training **RoBERTa-large** was computationally expensive.
- **Solution:** Used **Low-Rank Adaptation (LoRA)** to fine-tune only a subset of parameters, reducing memory usage while maintaining high performance.

🔹 **Dataset Preprocessing Issues:**

- **Challenge:** The IMDB dataset required extensive text cleaning and handling of imbalanced classes.
- **Solution:** Implemented **NLTK-based preprocessing** (lemmatization, stopword removal) and **balanced batches** using PyTorch DataLoader.

🔹 **Hyperparameter Optimization:**

- **Challenge:** Finding the best learning rate, batch size, and hidden dimensions was time-intensive.
- **Solution:** Used **Grid Search** to optimize hyperparameters efficiently.