import streamlit as st
import pandas as pd
import pickle


@st.cache_resource
def load_artifacts():
    with open('movies.pkl', 'rb') as f:
        movies_df = pickle.load(f)
    
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
        
    return movies_df, similarity

movies_df, similarity = load_artifacts()


def recommend(movie):
    index = movies_df[movies_df['original_title'] == movie].index[0]
    
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    

    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(movies_df.iloc[i[0]].original_title)
        
    return recommended_movie_names

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation Engine")
st.markdown("### *Content-Based Filtering via NLP Vectorization*")
st.markdown("---")
selected_movie_name = st.selectbox(
    "Choose a movie you like:",
    movies_df['original_title'].values
)

# Trigger recommendation
if st.button("Show Recommendations", use_container_width=True):
    with st.spinner('Computing vector similarity...'):
        recommendations = recommend(selected_movie_name)
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.success(recommendations[i])

st.markdown("---")
st.caption("Pre-computed similarity matrix serialized via Pickle | Inference decoupled from training | Memory-optimized caching via Streamlit")