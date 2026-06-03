import re
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Fragrance Recommender",
    layout="wide"
)

COLLECTION_FILE = Path("PerfumeCollection.csv")
DATABASE_FILE = Path("parfumo_data_clean.csv")

# Load data each time the app runs so the newest CSV rows are always available.
def load_data():
    collection = pd.read_csv(COLLECTION_FILE)
    database = pd.read_csv(DATABASE_FILE)
    return collection, database

# Save the fragrance collection back to the CSV file.
def save_collection(df: pd.DataFrame):
    df.to_csv(COLLECTION_FILE, index=False)

# Convert values to lowercase strings so keyword matching works reliably.
def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()

# Highlight five-star fragrances green in displayed tables.
def highlight_five_star_row(row):
    try:
        if int(row.get("Rating", 0)) == 5:
            return ["background-color: lightgreen"] * len(row)
    except Exception:
        pass
    return [""] * len(row)


def style_five_star(df: pd.DataFrame):
    if "Rating" in df.columns:
        return df.style.apply(highlight_five_star_row, axis=1)
    return df


def split_terms(text):
    if pd.isna(text) or not str(text).strip():
        return []
    raw = str(text)
    parts = re.split(r"[\,\|;/]+", raw)
    return [part.strip().lower() for part in parts if part.strip()]


def get_top_terms(series: pd.Series, top_n=10):
    terms = []
    for value in series.dropna():
        terms.extend(split_terms(value))
    if not terms:
        return pd.Series(dtype=int)
    return pd.Series(terms).value_counts().head(top_n)


def get_favorite_collection(collection: pd.DataFrame):
    wearable_mask = collection["Would I Wear?"].fillna("no").str.strip().str.lower() != "no"
    favorite = collection[wearable_mask & (collection["Rating"] >= 4)]
    if favorite.empty:
        favorite = collection[wearable_mask]
    return favorite


def explain_recommendations(results: pd.DataFrame, query_terms):
    rows = []
    for _, row in results.head(10).iterrows():
        matches = []
        for field in ["Themes", "Strength", "Season", "Gender", "Notes"]:
            value = str(row.get(field, ""))
            if any(term in value.lower() for term in query_terms):
                matches.append(field)
        if not matches:
            matches = ["Matches your preferred collection profile"]
        rows.append({
            "Name": row.get("Name", ""),
            "Brand": row.get("Brand", ""),
            "Reason": ", ".join(matches)
        })
    return pd.DataFrame(rows)


def bold_search_terms(text, query_terms):
    if pd.isna(text) or not query_terms:
        return text
    text = str(text)
    for term in sorted(set(query_terms), key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(lambda m: f"<b>{m.group(0)}</b>", text)
    return text


def style_discover_matches(df: pd.DataFrame, query_terms):
    styler = style_five_star(df)
    if query_terms:
        formatters = {
            col: lambda v, terms=query_terms: bold_search_terms(v, terms)
            for col in ["Name", "Brand", "Main_Accords", "Top_Notes", "Middle_Notes", "Base_Notes"]
            if col in df.columns
        }
        if formatters:
            styler = styler.format(formatters, escape=False)
    return styler

# Score how well a row matches the user's query terms.
def build_match_score(row, query_terms, include_name_brand=True):
    fields = []
    if include_name_brand:
        fields.extend([
            normalize_text(row.get("Name", "")),
            normalize_text(row.get("Brand", ""))
        ])
    fields.extend([
        normalize_text(row.get("Themes", "")),
        normalize_text(row.get("Strength", "")),
        normalize_text(row.get("Season", "")),
        normalize_text(row.get("Gender", "")),
        normalize_text(row.get("Notes", "")),
        normalize_text(row.get("Would I Wear?", ""))
    ])
    searchable_text = " ".join(fields)

    score = 0
    for term in query_terms:
        if term in searchable_text:
            score += 1
    return score

def semantic_recommend_from_collection(collection: pd.DataFrame, query: str, min_rating: int):
    wearable_mask = collection["Would I Wear?"].fillna("no").str.strip().str.lower() != "no"
    filtered = collection[wearable_mask & (collection["Rating"] >= min_rating)].copy()
    if filtered.empty:
        return filtered

    if not query.strip():
        return filtered.sort_values("Rating", ascending=False)

    documents = []
    for _, row in filtered.iterrows():
        documents.append(" ".join([
            normalize_text(row.get("Themes", "")),
            normalize_text(row.get("Strength", "")),
            normalize_text(row.get("Season", "")),
            normalize_text(row.get("Gender", "")),
            normalize_text(row.get("Notes", ""))
        ]))

    query_text = normalize_text(query)
    corpus = documents + [query_text]
    vectorizer = TfidfVectorizer().fit(corpus)
    vectors = vectorizer.transform(corpus)

    similarity_scores = cosine_similarity(vectors[:-1], vectors[-1:]).flatten()
    filtered["Semantic Score"] = similarity_scores
    filtered["Combined Score"] = filtered["Semantic Score"] * 0.7 + (filtered["Rating"] / 5.0) * 0.3
    filtered = filtered.sort_values(["Combined Score", "Rating"], ascending=[False, False])
    return filtered

# Recommend fragrances from your personal collection.
def recommend_from_collection(collection: pd.DataFrame, query: str, min_rating: int):
    wearable_mask = collection["Would I Wear?"].fillna("no").str.strip().str.lower() != "no"

    if not query.strip():
        return collection[wearable_mask & (collection["Rating"] >= min_rating)].sort_values("Rating", ascending=False)

    query_terms = [term.strip().lower() for term in query.split(",") if term.strip()]
    scores = [build_match_score(row, query_terms, include_name_brand=False) for _, row in collection.iterrows()]

    results = collection.copy()
    results["Match Score"] = scores
    results = results[
        wearable_mask &
        (results["Match Score"] > 0) &
        (results["Rating"] >= min_rating)
    ]
    results = results.sort_values(["Match Score", "Rating"], ascending=[False, False])

    return results

def tokenize_terms(value):
    value = normalize_text(value)
    if not value:
        return set()

    parts = re.split(r"[,\|;/]+", value)
    terms = set()
    for part in parts:
        for token in part.strip().split():
            clean = token.strip()
            if clean:
                terms.add(clean)
    return terms


def get_liked_collection_terms(collection: pd.DataFrame):
    wearable_mask = collection["Would I Wear?"].fillna("no").str.strip().str.lower() != "no"
    liked = collection[wearable_mask & (collection["Rating"] >= 4)]
    if liked.empty:
        liked = collection[wearable_mask]

    terms = set()
    for _, row in liked.iterrows():
        terms |= tokenize_terms(" ".join([
            row.get("Name", ""),
            row.get("Brand", ""),
            row.get("Themes", ""),
            row.get("Notes", "")
        ]))
    return terms


# Recommend new fragrances from the Parfumo database based on what you already like.
def recommend_from_database(database: pd.DataFrame, collection: pd.DataFrame, query: str):
    liked_terms = get_liked_collection_terms(collection)
    if not liked_terms and not query.strip():
        return database.head(20)

    query_terms = [term.strip().lower() for term in query.split(",") if term.strip()]
    scores = []

    for _, row in database.iterrows():
        searchable_text = " ".join([
            normalize_text(row.get("Name", "")),
            normalize_text(row.get("Brand", "")),
            normalize_text(row.get("Main_Accords", "")),
            normalize_text(row.get("Top_Notes", "")),
            normalize_text(row.get("Middle_Notes", "")),
            normalize_text(row.get("Base_Notes", ""))
        ])

        score_liked = sum(1 for term in liked_terms if term in searchable_text)
        score_query = sum(1 for term in query_terms if term in searchable_text)
        scores.append(score_liked * 2 + score_query)

    results = database.copy()
    results["Similarity Score"] = scores
    results = results[results["Similarity Score"] > 0]
    results = results.sort_values(["Similarity Score", "Rating_Value", "Rating_Count"], ascending=[False, False, False])

    return results

collection, database = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Fragrance Input", "Wear Today", "Discover", "Analytics"])

with tab1:
    st.header("Add a new fragrance to your collection")
    st.write("Enter the fragrance details for every column in your fragrance collection CSV.")

    left, right = st.columns(2)
    with left:
        name = st.text_input("Fragrance Name")
        brand = st.text_input("Brand")
        rating_label = st.radio(
            "My Rating",
            ["★", "★★", "★★★", "★★★★", "★★★★★"],
            index=2
        )
        rating = len(rating_label)
        review_on_parfumo = st.checkbox("Review on Parfumo?", value=False)
        wear = st.selectbox("Would I Wear?", ["Yes", "No"])
        bottle = st.selectbox("Sample/Full Bottle?", ["Sample", "Full Bottle"])

    with right:
        themes = st.text_input("Themes / Vibe (comma separated)")
        strength = st.selectbox("Strength", ["Unknown", "Subtle", "Normal", "Strong", "Intense"])
        season = st.selectbox("Season", ["Any", "Spring", "Summer", "Autumn", "Winter"])
        gender = st.selectbox("Gender", ["Any", "Feminine", "Masculine", "Unisex"])
        notes = st.text_area("Notes / Key accords (comma separated)")

    if st.button("Add fragrance"):
        new_row = {
            "Name": name.strip(),
            "Brand": brand.strip(),
            "Rating": rating,
            "Review on Parfumo?": review_on_parfumo,
            "Would I Wear?": wear,
            "Sample/Full Bottle?": bottle,
            "Themes": themes.strip(),
            "Strength": strength,
            "Season": season,
            "Gender": gender,
            "Notes": notes.strip()
        }

        collection = pd.concat([collection, pd.DataFrame([new_row])], ignore_index=True)
        save_collection(collection)

        st.success(f"Added '{name}' to your collection.")
        st.dataframe(style_five_star(collection.tail(10)))

with tab2:
    st.header("Wear Today")
    st.write("Describe how you feel or the kind of vibe you want, and the app will suggest fragrances from your collection.")

    vibe = st.text_input("Tell me your mood, theme, or note (for example: cozy, woody, spring, floral)")
    min_rating_label = st.radio(
        "Minimum rating to include",
        ["★", "★★", "★★★", "★★★★", "★★★★★"],
        index=2
    )
    min_rating = len(min_rating_label)

    if st.button("Recommend from my collection"):
        results = semantic_recommend_from_collection(collection, vibe, min_rating)
        if results.empty:
            st.info("No matches found. Try a broader search like 'vanilla', 'fresh', or 'cosy'.")
        else:
            st.subheader("Best matches from your collection")
            display = results.drop(columns=["Match Score"], errors="ignore")
            st.dataframe(style_five_star(display))

with tab3:
    st.header("Discover")
    st.write("Search the Parfumo database for new fragrances by brand, note, accord, or vibe.")

    query = st.text_input("Search by notes, accords, brand, or fragrance style")

    if st.button("Recommend new fragrances"):
        results = recommend_from_database(database, collection, query)
        if results.empty:
            st.info("No new fragrances found. Try shorter keywords like 'rose', 'oud', 'amber', or 'fresh'.")
        else:
            st.subheader("New fragrance recommendations")
            query_terms = [term.strip().lower() for term in query.split(",") if term.strip()]
            st.dataframe(style_discover_matches(
                results.loc[:, [
                    "Name",
                    "Brand",
                    "Release_Year",
                    "Concentration",
                    "Main_Accords",
                    "Top_Notes",
                    "Middle_Notes",
                    "Base_Notes",
                    "Rating_Value",
                    "Rating_Count",
                    "URL"
                ]].head(30),
                query_terms=query_terms
            ))

with tab4:
    st.header("Analytics")
    st.write("Insights based on your current fragrance collection.")

    favorites = get_favorite_collection(collection)
    favorite_notes = get_top_terms(favorites["Notes"], top_n=15)
    favorite_themes = get_top_terms(favorites["Themes"], top_n=15)
    favorite_brands = favorites["Brand"].value_counts().head(15)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Favourite notes")
        if not favorite_notes.empty:
            st.write(", ".join(favorite_notes.index[:10]))
        else:
            st.write("No notes available yet.")
    with col2:
        st.subheader("Favourite themes")
        if not favorite_themes.empty:
            st.write(", ".join(favorite_themes.index[:10]))
        else:
            st.write("No themes available yet.")
    with col3:
        st.subheader("Favourite brands")
        if not favorite_brands.empty:
            st.write(", ".join(favorite_brands.index[:10]))
        else:
            st.write("No brands available yet.")

    st.subheader("Note frequency")
    if not favorite_notes.empty:
        st.bar_chart(favorite_notes)
    else:
        st.info("Add fragrances with note data to generate a note frequency chart.")

    st.subheader("Theme frequency")
    if not favorite_themes.empty:
        st.bar_chart(favorite_themes)
    else:
        st.info("Add fragrances with theme data to generate a theme frequency chart.")

    st.subheader("Recommendation explanations")
    explain_query = st.text_input("Why would I wear a fragrance for this mood / note?", "woody")
    if st.button("Explain recommendations"):
        explain_terms = [term.strip().lower() for term in explain_query.split(",") if term.strip()]
        explanation_results = recommend_from_collection(favorites, explain_query, 1)
        if explanation_results.empty:
            st.info("No recommendation explanations available for that search.")
        else:
            st.table(explain_recommendations(explanation_results, explain_terms))
