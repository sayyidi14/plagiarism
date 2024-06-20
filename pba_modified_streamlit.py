import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import streamlit as st

# Download stopwords dan tokenizer dari NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Inisialisasi stemmer dan stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    # Step 1: Cleaning data
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Step 2: Convert to lower case
    lower_case_text = cleaned_text.lower()
    
    # Step 3: Tokenization
    tokens = word_tokenize(lower_case_text)
    
    # Step 4: Stopword removal
    tokens_without_stopwords = [word for word in tokens if word not in stop_words]
    
    # Step 5: Stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens_without_stopwords]
    
    # Join tokens back to string
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text


# Membaca dokumen dari file CSV
df = pd.read_csv('coba.csv')
documents = df['reviews'].tolist()


# Preprocessing dokumen
processed_docs = [preprocess_text(doc) for doc in documents]


# Mengubah teks menjadi fitur TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_docs)


# Pastikan `y` memiliki setidaknya dua kelas yang berbeda
y = df['label']

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Membuat pasangan dokumen untuk perbandingan
num_docs = len(documents)
pairs = [(i, j) for i in range(num_docs) for j in range(i + 1, num_docs)]


# Menghitung kesamaan kosinus untuk setiap pasangan
similarities = []
labels = []
threshold = 0.5

for (i, j) in pairs:
    similarity = cosine_similarity(X[i], X[j])[0][0]
    similarities.append(similarity)
    labels.append(1 if similarity > threshold else 0)


# Membagi data menjadi fitur (X) dan label (y)
X_pairs = np.array(similarities).reshape(-1, 1)
y_pairs = np.array(labels)


# Melatih model SVM
model_svm = SVC(kernel='linear')
model_svm.fit(X_pairs, y_pairs)


# Fungsi untuk mendeteksi plagiarisme antar dokumen baru menggunakan SVM
def detect_plagiarism_svm(sentence, docs):
    processed_sentence = preprocess_text(sentence)
    processed_docs = [preprocess_text(doc) for doc in docs]
    
    all_docs = processed_docs + [processed_sentence]
    X_all = vectorizer.transform(all_docs)
    
    new_similarities = [cosine_similarity(X_all[i], X_all[-1])[0][0] for i in range(len(docs))]
    
    new_X_pairs = np.array(new_similarities).reshape(-1, 1)
    predictions = model_svm.predict(new_X_pairs)
    
    return predictions


# Fungsi untuk mendeteksi plagiarisme menggunakan model SVM
def detect_plagiarism_svm(sentence, documents):
    # Preprocess the new sentence
    processed_sentence = preprocess_text(sentence)
    
    # Transform the processed sentence to TF-IDF vector
    sentence_vector = vectorizer.transform([processed_sentence])
    
    # Predict similarity with each document
    similarities = cosine_similarity(sentence_vector, X)
    
    # Binarize the similarity results
    threshold = 0.5  # You can adjust the threshold based on your needs
    predictions = (similarities >= threshold).astype(int).flatten()
    
    return predictions

st.markdown("<h1 style='text-align: center;'>Plagiarism Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Plagiarism Checker Detects Plagiarism in your Text</p>", unsafe_allow_html=True)

# Contoh penggunaan deteksi plagiarisme
new_sentence = st.text_area("",
        placeholder= "Input text here",  height=250
    )
st.write(f"You wrote {len(new_sentence)} characters.")
if st.button("Deteksi Menggunakan SVM"):
    # st.text("Deteksi menggunakan SVM:")
    predictions_svm = detect_plagiarism_svm(new_sentence, documents)
    similar_documents = [i for i, pred in enumerate(predictions_svm) if pred == 1]

    if similar_documents:
        for doc_idx in similar_documents:
            st.text(f"Kalimat tersebut mirip dengan dokumen {doc_idx+1}")
    else:
        st.text("Tidak ada dokumen yang mirip")



    # Evaluasi model SVM
    X_train, X_test, y_train, y_test = train_test_split(X_pairs, y_pairs, test_size=0.2, random_state=42)
    model_svm.fit(X_train, y_train)
    y_pred = model_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # st.text(f"\nAkurasi model SVM: {accuracy * 100:.2f}%")
    st.success(f"\nAkurasi model SVM: {accuracy * 100:.2f}%", icon="✅")