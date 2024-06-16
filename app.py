from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
import pickle
import ssl
import os

# Bypass SSL verification for NLTK data download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/story_db'
db = SQLAlchemy(app)

# Ensure the model and vectorizer paths are correct
# Update with the correct path
model_path = "/Users/erikprakoso/Work/Personal/project/flask/story-backend/rekomendasiByStoryID.h5"
# Update with the correct path
vectorizer_path = "/Users/erikprakoso/Work/Personal/project/flask/story-backend/vectorizer.pkl"

# Load the pre-trained model
try:
    model = load_model(model_path)
    print("Model has been successfully loaded.")
except OSError as e:
    print(f"Error: {e}")

# Load the vectorizer
try:
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer has been successfully loaded.")
except OSError as e:
    print(f"Error: {e}")

# Preprocess text function


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(tokens)

# Load and preprocess the dataset


def load_data():
    try:
        engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
        df_story = pd.read_sql('SELECT * FROM stories', engine)
        df_story['combined_text'] = df_story.apply(lambda row: ' '.join([
            row['overview'],
            row['author'],
            row['origin'],
            row['genre']
        ]), axis=1)
        df_story['combined_text'] = df_story['combined_text'].apply(
            preprocess_text)
        return df_story
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


df_story = load_data()
if df_story is None:
    print("Failed to load data from the database.")
    exit(1)


@app.route('/recommend', methods=['POST'])
def recommend_stories():
    story_id = request.json['story_id']
    top_n = request.json.get('top_n', 5)

    try:
        story_text = df_story[df_story['id'] ==
                              story_id]['combined_text'].values[0]
    except IndexError:
        return jsonify({"error": "Story ID not found."}), 404

    processed_story_text = preprocess_text(story_text)
    story_vector = vectorizer.transform([processed_story_text]).toarray()

    expected_shape = (1, vectorizer.max_features)
    if story_vector.shape[1] != expected_shape[1]:
        return jsonify({"error": f"Expected input shape {expected_shape}, but got {story_vector.shape}"}), 400

    predictions = model.predict(story_vector)
    recommended_story_indices = np.argsort(predictions[0])[-(top_n+1):][::-1]

    recommended_story_indices = [
        idx for idx in recommended_story_indices if idx < len(df_story)]

    if not recommended_story_indices:
        return jsonify({"error": "No valid recommendations found."}), 500

    recommended_stories = df_story.iloc[recommended_story_indices]
    recommended_stories = recommended_stories[recommended_stories['id']
                                              != story_id][:top_n]

    return recommended_stories.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
