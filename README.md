# Fake News Detection Using Machine Learning  

## Overview  
This project explores machine learning approaches to automatically classify the truthfulness of political statements using the **LIAR dataset**. Multiple feature extraction techniques and classification models were applied to distinguish between **true** and **fake** statements.  

## Objectives  
- Preprocess political statements for NLP tasks.  
- Extract features using **Bag of Words (BoW)**, **TF-IDF**, **Word2Vec**, and **GloVe** embeddings.  
- Train and evaluate models including **Logistic Regression**, **Support Vector Machines (SVM)**, and **Random Forest**.  
- Compare models based on accuracy, precision, recall, F1-score, and confusion matrix.  

## Dataset  
- **LIAR Dataset** – contains **12,836 short political statements** labeled into truth categories (e.g., pants-on-fire, false, half-true, mostly-true, true).  
- For this project, the labels were grouped into **binary classes**: `True` and `Fake`.  

## Methodology  
1. **Data Preprocessing**: Tokenization, stopword removal, and text cleaning.  
2. **Feature Extraction**:  
   - Bag of Words (BoW)  
   - TF-IDF  
   - Word2Vec embeddings  
   - Pre-trained GloVe embeddings  
3. **Machine Learning Models**:  
   - Logistic Regression  
   - SVM  
   - Random Forest  
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix  

## Results (Binary Classification)  
- **SVM + TF-IDF** gave the best performance with **~62% accuracy**.  
- Logistic Regression and Random Forest achieved comparable results (~59–61%).  
- Word embeddings (Word2Vec, GloVe) performed moderately well but less effective than TF-IDF.  

## Key Insights  
- **Traditional TF-IDF features** outperformed pre-trained embeddings in this dataset.  
- SVM proved to be the most effective model for binary classification.  
- Political statement classification remains a challenging task due to linguistic ambiguity.  

## Deliverables  
- **Dissertation Report**: Documenting methodology, experiments, and results.  
- **Presentation**: Summarizing the project for academic assessment.  

## Tools & Libraries  
- Python (NumPy, Pandas, scikit-learn)  
- NLTK / spaCy (text preprocessing)  
- Gensim (Word2Vec, GloVe)  
- Matplotlib / Seaborn (visualization)  
