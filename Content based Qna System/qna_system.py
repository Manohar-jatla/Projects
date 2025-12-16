import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt_tab')
nltk.download('stopwords') 

def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [stemmer.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
        processed_sentences.append(" ".join(words))  # Join words back into a sentence
    return sentences, processed_sentences # Return both original and processed sentences

def answer_question(paragraph, question):
    original_sentences, processed_sentences = preprocess_text(paragraph)
    _, processed_question = preprocess_text(question) 

    all_sentences = processed_sentences + processed_question
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    question_vector = tfidf_matrix[-1]  # The last vector is the question
    sentence_vectors = tfidf_matrix[:-1] # All sentence vectors

    similarity_scores = cosine_similarity(question_vector, sentence_vectors)
    
    most_similar_index = np.argmax(similarity_scores)

    return original_sentences[most_similar_index] # Return the original sentence

if __name__ == "__main__":
    paragraph = """
    CTC Tech is a technology solutions supplier based in Pune, India, specializing in delivering tailored IT facilities to businesses across various industries.
    With over 35 years of combined expertise, the company focuses on simplifying complex challenges through innovative solutions.
    They offer a wide range of services including custom software development, cloud migration, cybersecurity, data analytics, product design, and IT consulting.
    Its DataPulse platform helps organizations harness AI-driven data insights and predictive analytics to inform strategic decision-making.
    Tools like SmartAI Bot, Finance Buddy, and DataPulse, they empower businesses to leverage AI for conversational engagement, financial insights, and robust data analytics.
    The company aims to empower businesses to thrive in an ever-evolving digital landscape.
    """

    while True:
        print(paragraph)
        question = input("Ask a question about CTC Tech Services (or type 'exit'): ")
        if question.lower() == 'exit':
            break

        answer = answer_question(paragraph, question)
        print("Answer:", answer)
