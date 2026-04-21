import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, redirect, render_template, request, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
zomato_df = pd.read_csv('restaurant1.csv')

# Normalise column names once at startup
zomato_df['name_lower'] = zomato_df['name'].str.strip().str.lower()
zomato_df['cuisines'] = zomato_df['cuisines'].fillna('').astype(str)
zomato_df['cost'] = pd.to_numeric(zomato_df['cost'], errors='coerce').fillna(0)
zomato_df['Mean Rating'] = pd.to_numeric(zomato_df['Mean Rating'], errors='coerce').fillna(0)


# ==========================
# Recommendation Function
# ==========================
def get_recommendations(restaurant_name, cuisine, budget, rating):
    df = zomato_df.copy().reset_index(drop=True)

    # Normalise input
    restaurant_name_clean = restaurant_name.strip().lower()

    # Step 1: Find restaurant (case-insensitive, strip whitespace)
    match = df[df['name_lower'] == restaurant_name_clean]

    if match.empty:
        # Try partial match as fallback
        partial = df[df['name_lower'].str.contains(restaurant_name_clean, na=False)]
        if partial.empty:
            return "restaurant_not_found"
        match = partial

    # Use the first match index
    idx = match.index[0]

    # Step 2: TF-IDF on cuisines
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['cuisines'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get similarity scores for matched restaurant
    sim_scores = list(enumerate(cosine_sim[idx].flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude self; take top 50 candidates before filtering
    sim_indices = [i[0] for i in sim_scores if i[0] != idx][:50]

    if not sim_indices:
        return "no_recommendations"

    recommendations = df.iloc[sim_indices].copy()

    # Step 3: Apply filters safely
    try:
        budget_val = float(budget)
    except (TypeError, ValueError):
        budget_val = 9999999

    try:
        rating_val = float(rating)
    except (TypeError, ValueError):
        rating_val = 0.0

    cuisine_clean = cuisine.strip() if cuisine else ''

    filtered = recommendations.copy()

    if cuisine_clean:
        filtered = filtered[filtered['cuisines'].str.contains(cuisine_clean, case=False, na=False)]

    filtered = filtered[
        (filtered['cost'] <= budget_val) &
        (filtered['Mean Rating'] >= rating_val)
    ]

    # If strict filter returns nothing, relax cuisine filter
    if filtered.empty and cuisine_clean:
        filtered = recommendations[
            (recommendations['cost'] <= budget_val) &
            (recommendations['Mean Rating'] >= rating_val)
        ]

    if filtered.empty:
        return "no_matches"

    # Remove duplicates, sort by rating
    filtered = filtered.drop_duplicates(subset=['name'])
    filtered = filtered.sort_values(by='Mean Rating', ascending=False)

    return filtered.head(10)


# ==========================
# Routes
# ==========================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


@app.route('/result', methods=['POST'])
def result():
    restaurant_name = request.form.get('restaurant_name', '').strip()
    cuisine = request.form.get('cuisine', '')
    budget = request.form.get('budget', '9999999')
    rating = request.form.get('rating', '0')

    if not restaurant_name:
        return render_template('error.html',
                               message="Please enter a restaurant name to search.",
                               show_back=True)

    top_restaurants = get_recommendations(restaurant_name, cuisine, budget, rating)

    if isinstance(top_restaurants, str):
        messages = {
            "restaurant_not_found": f"No restaurant named \"{restaurant_name}\" was found in our database. Please check the spelling or try a different name.",
            "no_recommendations": "We couldn't generate recommendations for this restaurant.",
            "no_matches": "No restaurants match your selected filters. Try relaxing your budget or rating criteria."
        }
        return render_template("error.html",
                               message=messages.get(top_restaurants, top_restaurants),
                               show_back=True)

    top_restaurants_list = top_restaurants.to_dict('records')
    return render_template('result.html',
                           recommended_restaurants=top_restaurants_list,
                           search_name=restaurant_name)


@app.route('/analytics')
def analytics():
    import matplotlib.pyplot as plt
    zomato_df['Mean Rating'].hist()
    plt.savefig('static/ratings.png')
    return render_template('analytics.html')


if __name__ == '__main__':
    app.run(debug=True)