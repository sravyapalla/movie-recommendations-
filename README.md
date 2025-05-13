# Movie Recommendation System

This project builds a movie recommendation system using the MovieLens 100k dataset and `scikit-surprise` for collaborative filtering with SVD. It includes:

- A Jupyter Notebook (`movie_recommendation.ipynb`) with the model training and an `ipywidgets` frontend.
- A Flask web app (`app.py` and `templates/index.html`) for a browser-based interface.

## Setup
1. Install dependencies: `pip install pandas scikit-surprise ipywidgets flask`
2. Run the notebook in Jupyter: `jupyter notebook`
3. Run the Flask app: `python app.py`

## Dataset
- `u.data`: User ratings
- `u.item`: Movie metadata
