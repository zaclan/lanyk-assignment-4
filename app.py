from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

stop_words = stopwords.words('english')

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)
doc_term_matrix = vectorizer.fit_transform(documents)

# Apply SVD (LSA)
svd = TruncatedSVD(n_components=100)  # You can adjust n_components as needed
lsa_matrix = svd.fit_transform(doc_term_matrix)


def search_engine(query):
    try:
        print(f"Processing query: {query}")

        # Preprocess the query
        query_vector = vectorizer.transform([query])
        print(f"Query vector shape: {query_vector.shape}")

        if query_vector.nnz == 0:
            print("Query vector is empty after removing stopwords.")
            return [], [], []

        query_lsa = svd.transform(query_vector)
        print(f"Query LSA shape: {query_lsa.shape}")

        # Compute cosine similarities
        similarities = cosine_similarity(query_lsa, lsa_matrix)[0]
        print(f"Similarities calculated. Shape: {similarities.shape}")

        # Get top 5 documents
        top_indices = similarities.argsort()[-5:][::-1]
        top_similarities = similarities[top_indices]
        top_documents = [documents[i] for i in top_indices]

        return top_documents, top_similarities.tolist(), top_indices.tolist()
    except Exception as e:
        print(f"Error in search_engine: {e}")
        return [], [], []



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
