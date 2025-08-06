import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Load your dataset (replace with actual path to your movies.csv)
movies_data = pd.read_csv("movies.csv")

# Handle missing values
for column in movies_data.select_dtypes(include=['object']).columns:
    movies_data[column] = movies_data[column].fillna('')

# Combine selected features into one string
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
movies_data['combined_features'] = movies_data[selected_features].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF Vectorization and Similarity
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])
similarity = cosine_similarity(feature_vectors)

# KMeans Clustering
k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
movies_data['cluster'] = kmeans.fit_predict(feature_vectors)

# Lists for dropdowns
list_of_all_titles = movies_data['title'].tolist()
genre_list = sorted(set(g.strip() for sublist in movies_data['genres'].str.split('|') for g in sublist if g))

# Mood mapping
mood_mapping = {
    'Happy': ['Comedy', 'Romance', 'Musical'],
    'Emotional': ['Drama', 'Biography', 'Family'],
    'Thrilling': ['Action', 'Thriller', 'Adventure'],
    'Spooky': ['Horror', 'Mystery', 'Supernatural'],
    'Fantasy': ['Sci-Fi', 'Fantasy', 'Animation']
}

# ================= TKINTER GUI =================
root = tk.Tk()
root.title('Movie Recommender System')
root.geometry('800x700')
root.configure(bg='#f0f0f0')


def recommend_movies():
    movie_name = search_entry.get()
    if movie_name:
        match = difflib.get_close_matches(movie_name, list_of_all_titles)
        if match:
            close_match = match[0]
            index = movies_data[movies_data.title == close_match].index[0]
            similarity_score = list(enumerate(similarity[index]))
            sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:11]
            result_box.delete(0, tk.END)
            for movie in sorted_movies:
                result_box.insert(tk.END, movies_data.iloc[movie[0]]['title'])
        else:
            messagebox.showinfo('Not Found', 'No close match found.')
    else:
        messagebox.showinfo('Error', 'Please enter a movie name.')

def recommend_by_mood():
    mood = mood_combo.get()
    if mood != 'Select Mood':
        genres = mood_mapping[mood]
        mood_movies = movies_data[movies_data['genres'].apply(lambda x: any(g in x for g in genres))]
        result_box.delete(0, tk.END)
        for _, row in mood_movies.head(10).iterrows():
            result_box.insert(tk.END, row['title'])
    else:
        messagebox.showinfo('Error', 'Select a mood.')

def recommend_by_genre_cluster():
    genre = genre_combo.get()
    if genre != 'Select Genre':
        genre_movies = movies_data[movies_data['genres'].str.contains(genre, case=False, na=False)]
        if not genre_movies.empty:
            cluster = genre_movies.iloc[0]['cluster']
            clustered = movies_data[(movies_data['cluster'] == cluster) & (movies_data['genres'].str.contains(genre, case=False))]
            result_box.delete(0, tk.END)
            for _, row in clustered.head(10).iterrows():
                result_box.insert(tk.END, row['title'])
        else:
            messagebox.showinfo('No Movies', 'No movies found in this genre.')
    else:
        messagebox.showinfo('Error', 'Select a genre.')

def reset_fields():
    search_entry.delete(0, tk.END)
    mood_combo.set('Select Mood')
    genre_combo.set('Select Genre')
    result_box.delete(0, tk.END)

def plot_similarity_heatmap():
    indices = list(range(min(10, len(movies_data))))
    sim_matrix = similarity[indices, :][:, indices]
    labels = movies_data.iloc[indices]['title']

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap='coolwarm')
    plt.title('Cosine Similarity Heatmap (Top 10 Movies)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_kmeans_clusters():
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(feature_vectors.toarray())

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=movies_data['cluster'], cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title('KMeans Clustering (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.show()

def plot_elbow_method():
    inertias = []
    for k in range(1, 15):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(feature_vectors)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 15), inertias, marker='o')
    plt.title('Elbow Method - Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================= UI Widgets =================
tk.Label(root, text='Movie Recommender System', font=('Helvetica', 16), bg='#f0f0f0').pack(pady=10)

tk.Label(root, text='Enter Movie Name:', bg='#f0f0f0').pack()
search_entry = tk.Entry(root, width=50)
search_entry.pack(pady=5)
tk.Button(root, text='Recommend', command=recommend_movies, bg='#4CAF50', fg='white').pack(pady=5)

tk.Label(root, text='Or Choose a Mood:', bg='#f0f0f0').pack()
mood_combo = ttk.Combobox(root, values=['Select Mood'] + list(mood_mapping.keys()), state='readonly')
mood_combo.current(0)
mood_combo.pack(pady=5)
tk.Button(root, text='Recommend by Mood', command=recommend_by_mood, bg='#2196F3', fg='white').pack(pady=5)

tk.Label(root, text='Or Search by Genre (KMeans Clustering):', bg='#f0f0f0').pack(pady=(10, 0))
genre_combo = ttk.Combobox(root, values=['Select Genre'] + genre_list, state='readonly')
genre_combo.current(0)
genre_combo.pack(pady=5)
tk.Button(root, text='Recommend by Genre (KMeans)', command=recommend_by_genre_cluster, bg='#9C27B0', fg='white').pack(pady=5)

tk.Label(root, text='Recommendations:', bg='#f0f0f0').pack(pady=5)
result_box = tk.Listbox(root, height=10, width=60)
result_box.pack(pady=5)

tk.Button(root, text='Reset', command=reset_fields, bg='#f44336', fg='white').pack(pady=5)

# Graph Buttons
tk.Label(root, text='Visual Analysis:', bg='#f0f0f0').pack(pady=(20, 5))
tk.Button(root, text='Cosine Similarity Heatmap', command=plot_similarity_heatmap).pack(pady=2)
tk.Button(root, text='KMeans Clustering (PCA)', command=plot_kmeans_clusters).pack(pady=2)
tk.Button(root, text='Elbow Method (K Selection)', command=plot_elbow_method).pack(pady=2)

root.mainloop()