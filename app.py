from flask import Flask, render_template, request
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load and train the model
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', 
                     names=['movie_id', 'title', 'release_date', 'video_release', 'url'] + [f'genre_{i}' for i in range(19)])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
model = SVD(n_factors=100, random_state=42)
model.fit(trainset)

def get_top_n_recommendations(user_id, n=10):
    all_movie_ids = ratings['movie_id'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = [(pred.iid, pred.est) for pred in predictions[:n]]
    return top_n

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    user_id = None
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        if 1 <= user_id <= 943:
            recs = get_top_n_recommendations(user_id)
            recommendations = [(movies[movies['movie_id'] == movie_id]['title'].values[0], round(pred_rating, 2)) 
                              for movie_id, pred_rating in recs]
        else:
            recommendations = [("Invalid User ID", "Please enter a user ID between 1 and 943.")]
    return render_template('index.html', recommendations=recommendations, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)