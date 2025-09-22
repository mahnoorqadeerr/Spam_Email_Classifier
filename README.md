# 📧 Email Spam Classifier

A complete machine learning project that classifies SMS messages as spam or not spam using Python, scikit-learn, and NLTK. This project includes data preprocessing, exploratory data analysis (EDA), feature extraction, model training, and evaluation using multiple classifiers.

---

## 🎯 Project Goal

Predict whether a message is spam based on its content using supervised learning techniques. The project compares multiple models and evaluates them using precision, recall, and F1-score for deeper insight.

---

## 📁 Dataset

- Public SMS spam dataset with labeled messages  
- Labels: `1 = spam`, `0 = ham (not spam)`

---

## 🧪 Workflow Summary

### 🔹 Data Preprocessing
- Removed punctuation and special characters  
- Converted text to lowercase  
- Removed stopwords using NLTK  
- Mapped labels to binary format

### 🔹 Exploratory Data Analysis (EDA)
- Bar chart showing spam vs. non-spam distribution  
- Word clouds for spam and ham messages  
- Comparison of average message lengths

### 🔹 Feature Extraction
- Used `CountVectorizer` to convert text into numerical features

### 🔹 Model Training & Evaluation
Trained and compared three models:
- **Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## 📊 Results Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Naive Bayes        | 0.99     | 0.97      | 0.95   | 0.96     |
| Logistic Regression| 0.99     | 1.00      | 0.89   | 0.94     |
| SVM                | 0.99     | 1.00      | 0.89   | 0.94     |

---

## 🧠 Conclusion

Naive Bayes achieved the highest recall, making it best for catching spam. Logistic Regression and SVM had perfect precision, minimizing false positives. This comparison highlights the importance of choosing models based on task-specific priorities.

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- scikit-learn  
- NLTK  
- Matplotlib, Seaborn  

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn nltk wordcloud matplotlib seaborn

# Upload dataset and run all cells in the notebook

```
## 🔮 Future Work
- Try deep learning models like LSTM or BERT for improved performance.
- Deploy the model using Streamlit or Flask for real-time spam detection.
- Add cross-validation and hyperparameter tuning for better generalization.

## 👩‍💻 Author

**Mahnoor Qadeer**  
Aspiring AI student passionate about NLP, ethical machine learning, and building real-world projects for global impact.
