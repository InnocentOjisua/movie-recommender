import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
import pickle

# 1️ Load raw CSVs
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")

# 2️ Merge credits with movies
credits_renamed = credits.rename(columns={"movie_id": "id"})
movies_merged = movies.merge(credits_renamed, on='id')

# 3️ Clean the dataset
movies_cleaned = movies_merged.drop(columns=[
    'homepage', 'title_x', 'title_y', 'status', 'production_countries'
])

# Combine overview + plot for richer recommendations
if 'plot' in movies_cleaned.columns:
    movies_cleaned['full_text'] = movies_cleaned['overview'].fillna('') + " " + movies_cleaned['plot'].fillna('')
else:
    movies_cleaned['full_text'] = movies_cleaned['overview'].fillna('')

# 4️ Save cleaned dataframe for later use
movies_cleaned.to_csv('movies_cleaned.csv', index=False)

# 5️ Compute TF-IDF matrix
tfv = TfidfVectorizer(
    min_df=3,
    max_features=3000, 
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    stop_words='english'
)
tfv_matrix = tfv.fit_transform(movies_cleaned['full_text'])

# Compute sparse cosine similarity
sig = cosine_similarity(tfv_matrix, tfv_matrix, dense_output=False)

# Save sparse matrix and indices mapping
save_npz('sig_matrix.npz', sig)

indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()
with open('indices.pkl', 'wb') as f:
    pickle.dump(indices, f)

print("✅ Preprocessing complete! Files saved: movies_cleaned.csv, sig_matrix.npz, indices.pkl")
