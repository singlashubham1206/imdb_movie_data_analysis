import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("movies_cleaned.csv")

    # Convert multi-value columns to lists
    cols_to_split = ['Genres Categories', 'Directors', 'Writers', 'Stars']
    for col in cols_to_split:
        df[col] = df[col].astype(str).apply(lambda x: x.split('|') if pd.notna(x) else [])

    # Trim spaces
    for col in cols_to_split:
        df[col] = df[col].apply(lambda x: [s.strip() for s in x])

    # Compute Weighted Score (Alternative: IMDb Formula)
    df['Weighted Score'] = df['Rating'] * df['Vote Count (M)']

    return df

df = load_data()

# # Sidebar - Filters
st.sidebar.header("Filters 🔍")
selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + sorted(set(sum(df['Genres Categories'], []))))
selected_director = st.sidebar.selectbox("Select Director", ["All"] + sorted(set(sum(df['Directors'], []))))
selected_decade = st.sidebar.selectbox("Select Decade", ["All"] + sorted(df['Decade'].unique()))

# Filter Data
filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['Genres Categories'].apply(lambda x: selected_genre in x)]
if selected_director != "All":
    filtered_df = filtered_df[filtered_df['Directors'].apply(lambda x: selected_director in x)]
if selected_decade != "All":
    filtered_df = filtered_df[filtered_df['Decade'] == selected_decade]

# Title
st.title("🎬 Movie Insights Dashboard")


# Top-Rated Movies
st.subheader("⭐ Top-Rated Movies")
top_movies = filtered_df[['Title', 'Rating', 'Vote Count (M)', 'Weighted Score']].sort_values(
    by='Weighted Score', ascending=False).head(10).reset_index(drop=True)
# Format numeric columns to display 1 decimal place
top_movies['Rating'] = top_movies['Rating'].map(lambda x: f"{x:.1f}")
top_movies['Vote Count (M)'] = top_movies['Vote Count (M)'].map(lambda x: f"{x:.1f}  Million")
top_movies['Weighted Score'] = top_movies['Weighted Score'].map(lambda x: f"{x:.0f}")
top_movies = top_movies.rename(columns={'Weighted Score': 'Weighted Score ⬇️'})
st.table(top_movies)
st.write("")  


# Bar Chart: Top 5 Directors
st.subheader("🎥 Top 5 Directors by Number of Movies")
df_directors = filtered_df.explode('Directors')
top_directors = df_directors['Directors'].value_counts().head(5)
fig, ax = plt.subplots()
sns.barplot(x=top_directors.values, y=top_directors.index, palette='viridis', ax=ax)
ax.set_xlabel("Number of Movies")
ax.set_ylabel("Director")
ax.set_title("Top 5 Directors")
st.pyplot(fig)
st.write("") 

# Bar Chart: Top 5 Actors
st.subheader("🎭 Top 5 Actors by Number of Movies")
df_actors = filtered_df.explode('Stars')
top_actors = df_actors['Stars'].value_counts().head(5)
fig, ax = plt.subplots()
sns.barplot(x=top_actors.values, y=top_actors.index, palette='magma', ax=ax)
ax.set_xlabel("Number of Movies")
ax.set_ylabel("Actor")
ax.set_title("Top 5 Actors")
st.pyplot(fig)
st.write("")  


# Genre Distribution
st.subheader("🍿 Genre Distribution")

# Explode the list column to get individual genres
df_genres = filtered_df.explode('Genres Categories')

genre_counts = df_genres['Genres Categories'].value_counts()
threshold = 0.04 * genre_counts.sum() 

large_genres = genre_counts[genre_counts >= threshold]
small_genres = genre_counts[genre_counts < threshold]

# Add "Other" category if there are small genres
if not small_genres.empty:
    large_genres['Other'] = small_genres.sum()

# Generate a pastel color palette
colors = sns.color_palette('pastel', len(large_genres) - 1)
colors.append('#808080')

# Plot the pie chart
fig, ax = plt.subplots()
ax.pie(
    large_genres, 
    labels=large_genres.index, 
    autopct='%1.1f%%', 
    colors=colors
)
ax.set_title("Genres")

# Display in Streamlit
st.pyplot(fig)
st.write("")  


# Line Chart: Movies Released Per Decade
st.subheader("📅 Movies Released Per Decade")
df_decades = filtered_df.groupby('Decade').size()
fig, ax = plt.subplots()
df_decades.plot(kind='line', marker='o', color='blue', ax=ax)
ax.set_xlabel("Decade")
ax.set_ylabel("Number of Movies")
ax.set_title("Movies Released Over Time")
st.pyplot(fig)
st.write("") 

# Genre Trends Over Decades
st.subheader("📈 Genre Trends Over Decades")

# Explode genres into separate rows
df_exploded_genres = filtered_df.explode('Genres Categories')

# Identify top 5 genres
top_genres = df_exploded_genres['Genres Categories'].value_counts().head(5)
top_genres_list = top_genres.index

# Count number of movies per genre per decade
genre_trends = df_exploded_genres.groupby(['Decade', 'Genres Categories']).size().unstack().fillna(0)

# Plot trends for the top genres
fig, ax = plt.subplots(figsize=(12, 6))

# Use seaborn color palette for better aesthetics
colors = sns.color_palette("husl", len(top_genres_list))

# Plot each genre with customized styling
for genre, color in zip(top_genres_list, colors):
    ax.plot(genre_trends.index, genre_trends[genre], marker='o', linestyle='-', linewidth=2, markersize=8, label=genre, color=color)

# Beautify plot
ax.set_title("📊 Genre Trends Over Decades", fontsize=16, fontweight='bold', color="#333")
ax.set_xlabel("Decade", fontsize=13)
ax.set_ylabel("Number of Movies", fontsize=13)
ax.legend(title="Genres", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.7)

# **Skip every other decade label on X-axis**
ax.set_xticks(genre_trends.index[::2])  # This will only show every second label

# Display in Streamlit
st.pyplot(fig)

# # Create the figure and plot
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.histplot(df['Duration (Mins)'], bins=30, kde=True, ax=ax)

# # Labels and title
# ax.set_title("Distribution of Movie Durations")
# ax.set_xlabel("Duration (Minutes)")
# ax.set_ylabel("Frequency")

# # Display in Streamlit
# st.subheader("⏳ Movie Duration Distribution")
# st.pyplot(fig)

# movie Duration Distribution
st.subheader("⏳ Movie Duration Distribution")

# Create Altair histogram
hist_chart = alt.Chart(df).mark_bar().encode(
    alt.X('Duration (Mins)', bin=alt.Bin(maxbins=30), title="Duration (Minutes)"),
    alt.Y('count()', title="Frequency")
).properties(
    width=600,
    height=400,
    title="Distribution of Movie Durations"
)

st.altair_chart(hist_chart, use_container_width=True)
st.write("") 


# Movie Search with Autocomplete
st.subheader("🔍 Search for a Movie")

# Get unique movie titles
movie_titles = filtered_df['Title'].unique()

# Use a selectbox instead of text_input for suggestions
search_query = st.selectbox("Enter movie title:", [""] + sorted(movie_titles))

# Filter and show results
if search_query:
    search_results = filtered_df[filtered_df['Title'] == search_query]
    res = search_results[['Title', 'Duration (Mins)', 'Rating', 'Stars']]
    res['Rating'] = res['Rating'].map(lambda x: f"{x:.1f}")
    res['Stars'] = res['Stars'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    res.reset_index(drop=True)
    st.table(res)

st.write("") 


# st.subheader("📈 Genre Trends Over Decades")

# df_exploded_genres = filtered_df.explode('Genres Categories')

# top_genres = df_exploded_genres['Genres Categories'].value_counts().head(5)
# top_genres

# # Count number of movies per genre per decade
# genre_trends = df_exploded_genres.groupby(['Decade', 'Genres Categories']).size().unstack().fillna(0)

# # Plot trends for the top genres
# top_genres_list = top_genres.index  # Select top genres identified earlier
# fig, ax = plt.subplots(figsize=(12, 6))

# genre_trends[top_genres_list].plot(ax=ax, marker='o')

# ax.set_title("Genre Trends Over Decades")
# ax.set_xlabel("Decade")
# ax.set_ylabel("Number of Movies")
# ax.legend(title="Genres")
# ax.grid()

# st.pyplot(fig)




