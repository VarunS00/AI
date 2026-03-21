from flask import Flask, render_template, request
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movie data
with open("movie.json", "r") as f:
    movies = json.load(f)

# Build TF-IDF vectors for descriptions
descriptions = [m["description"] for m in movies]
tfidf = TfidfVectorizer(stop_words="english")
desc_vectors = tfidf.fit_transform(descriptions).toarray()

def similarity(i, liked_indices):
    score = 0
    for li in liked_indices:
        # Genre overlap
        score += len(set(movies[i]["genres"]) & set(movies[li]["genres"]))
        # Description similarity
        score += cosine_similarity([desc_vectors[i]], [desc_vectors[li]])[0][0]
    return score

@app.route("/")
def index():
    return render_template("index.html", movies=movies)

@app.route("/recommend", methods=["POST"])
def recommend():
    liked = request.form.getlist("liked")
    liked_indices = [int(i) for i in liked]

    scores = []
    for i in range(len(movies)):
        if i in liked_indices:
            continue
        scores.append((movies[i]["title"], similarity(i, liked_indices)))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:5]

    return render_template("results.html", recommendations=top)

if __name__ == "__main__":
    app.run(debug=True)