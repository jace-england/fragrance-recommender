import streamlit as st
import pandas as pd
import re
from collections import Counter

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Fragrance Recommender")

page = st.sidebar.radio(
    "Navigation",
    ["Wear Today", "Discover New Fragrances", "Analytics"]
)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_collection():
    return pd.read_csv("PerfumeCollection.csv")

@st.cache_data
def load_parfumo():
    df = pd.read_csv("parfumo_data_clean.csv")
    # Combine all note columns into one field
    def combine_notes(row):
        parts = [
            str(row["Top_Notes"]),
            str(row["Middle_Notes"]),
            str(row["Base_Notes"]),
        ]
        parts = [p for p in parts if p.lower() != "nan"]
        return ", ".join(parts)
    df["All_Notes"] = df.apply(combine_notes, axis=1)
    return df

collection = load_collection()
parfumo    = load_parfumo()

# ── HELPERS ────────────────────────────────────────────────────────────────────
def tokenize(text):
    """Return a set of lowercase word tokens from a string."""
    return set(re.findall(r"\b\w+\b", str(text).lower()))

def score_row_against_words(query_words, row_notes_words, row_theme_words,
                             row_season_words, row_strength_words, row_gender_words):
    """Return (score, match_info) for a collection row vs a set of query words."""
    score = 0
    note_matches     = []
    theme_matches    = []
    season_matches   = []
    strength_matches = []
    gender_matches   = []

    for word in query_words:
        if word in row_notes_words:
            score += 4
            note_matches.append(word)
        if word in row_theme_words:
            score += 2
            theme_matches.append(word)
        if word in row_season_words:
            score += 1
            season_matches.append(word)
        if word in row_strength_words:
            score += 1
            strength_matches.append(word)
        if word in row_gender_words:
            score += 1
            gender_matches.append(word)

    return score, {
        "notes":    note_matches,
        "themes":   theme_matches,
        "season":   season_matches,
        "strength": strength_matches,
        "gender":   gender_matches,
    }

def build_taste_profile(collection):
    """
    Learn the user's taste from fragrances they would wear.
    Returns weighted Counter dicts for notes and themes.
    """
    liked = collection[
        collection["Would I Wear?"].str.strip().str.lower() == "yes"
    ]

    note_counter  = Counter()
    theme_counter = Counter()

    for _, row in liked.iterrows():
        weight = float(row["Rating"]) / 5.0

        for word in tokenize(row["Notes"]):
            if len(word) > 2:          # skip tiny stopwords
                note_counter[word] += weight

        for word in tokenize(row["Themes"]):
            if len(word) > 2:
                theme_counter[word] += weight

    return note_counter, theme_counter

def render_match_card(row, match_info, rating_col="Rating",
                       name_col="Name", brand_col="Brand",
                       score_col="score", show_raw=True,
                       extra_fields=None):
    """Render a single recommendation card."""
    rating = row[rating_col]
    name   = row[name_col]
    brand  = row[brand_col]
    score  = row[score_col]

    if float(rating) == 5:
        st.markdown(
            f"<h2 style='color:#2E8B57'>{name}</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"## {name}")

    st.write(f"**Brand:** {brand}")
    st.write(f"**Rating:** {rating}")
    st.write(f"**Match Score:** {round(float(score), 1)}")

    if extra_fields:
        for label, value in extra_fields.items():
            st.write(f"**{label}:** {value}")

    # ── Why it matched summary ─────────────────────────────────────────────────
    reasons = []
    if match_info.get("notes"):
        reasons.append(f"contains notes: {', '.join(match_info['notes'])}")
    if match_info.get("themes"):
        reasons.append(f"matches themes: {', '.join(match_info['themes'])}")
    if match_info.get("season"):
        reasons.append(f"suitable for {', '.join(match_info['season'])}")
    if float(rating) >= 5:
        reasons.append("one of your 5-star fragrances")

    if reasons:
        st.success("Why it matched: " + "; ".join(reasons))

    if show_raw:
        if "Themes" in row.index:
            st.write(f"**Themes:** {row['Themes']}")
        if "Notes" in row.index:
            st.write(f"**Notes:** {row['Notes']}")

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# WEAR TODAY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Wear Today":

    st.title("Fragrance Recommendation Tool")
    st.write(f"You have **{len(collection)}** fragrances in your collection")

    col1, col2, col3 = st.columns(3)
    with col1:
        season_filter = st.selectbox(
            "Season", ["Any", "Spring", "Summer", "Autumn", "Winter"]
        )
    with col2:
        strength_filter = st.selectbox(
            "Strength", ["Any", "Light", "Moderate", "Strong"]
        )
    with col3:
        gender_filter = st.selectbox(
            "Gender", ["Any", "Masculine", "Feminine", "Unisex"]
        )

    query = st.text_input(
        "Describe what you're looking for today",
        placeholder="cosy tea woody autumn"
    )

    query_words = set(re.findall(r"\b\w+\b", query.lower())) if query else set()

    scores      = []
    match_infos = []

    for _, row in collection.iterrows():

        wear_flag = str(row["Would I Wear?"]).strip().lower()

        # ── Dropdown filters ───────────────────────────────────────────────────
        if (season_filter != "Any"
                and season_filter.lower() not in str(row["Season"]).lower()):
            scores.append(-1); match_infos.append({}); continue

        if (strength_filter != "Any"
                and strength_filter.lower() not in str(row["Strength"]).lower()):
            scores.append(-1); match_infos.append({}); continue

        if (gender_filter != "Any"
                and gender_filter.lower() not in str(row["Gender"]).lower()):
            scores.append(-1); match_infos.append({}); continue

        if wear_flag == "no":
            scores.append(-1); match_infos.append({}); continue

        # ── Tokenise fields ────────────────────────────────────────────────────
        notes_words    = tokenize(row["Notes"])
        theme_words    = tokenize(row["Themes"])
        season_words   = tokenize(row["Season"])
        strength_words = tokenize(row["Strength"])
        gender_words   = tokenize(row["Gender"])

        # ── No query: show all valid rows, sorted by rating ───────────────────
        if not query_words:
            scores.append(float(row["Rating"]))
            match_infos.append({
                "notes": [], "themes": [], "season": [],
                "strength": [], "gender": []
            })
            continue

        # ── Scored match ───────────────────────────────────────────────────────
        score, info = score_row_against_words(
            query_words, notes_words, theme_words,
            season_words, strength_words, gender_words
        )

        rating = float(row["Rating"])
        score  = score * (rating / 5)

        scores.append(score)
        match_infos.append(info)

    collection = collection.copy()
    collection["score"]      = scores
    collection["match_info"] = match_infos

    results = collection[collection["score"] > 0].sort_values(
        "score", ascending=False
    ).head(10)

    if results.empty:
        st.info("No matches found — try adjusting your filters or query.")
    else:
        st.subheader("Recommended")
        for _, row in results.iterrows():
            render_match_card(
                row,
                row["match_info"],
                rating_col="Rating",
                show_raw=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# DISCOVER NEW FRAGRANCES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Discover New Fragrances":

    st.title("Discover New Fragrances")
    st.write(
        "Based on what you love in your collection, "
        "here are fragrances from the Parfumo database you don't own yet."
    )

    # ── Learn taste from collection ────────────────────────────────────────────
    note_profile, theme_profile = build_taste_profile(collection)

    top_notes  = [w for w, _ in note_profile.most_common(20)]
    top_themes = [w for w, _ in theme_profile.most_common(10)]

    with st.expander("Your taste profile (learned from your collection)"):
        st.write(f"**Favourite notes:** {', '.join(top_notes)}")
        st.write(f"**Favourite themes / accords:** {', '.join(top_themes)}")

    # ── Owned name set for deduplication ──────────────────────────────────────
    owned_names = set(collection["Name"].str.strip().str.lower())

    # ── Optional extra query ───────────────────────────────────────────────────
    extra_query = st.text_input(
        "Refine further (optional)",
        placeholder="e.g. smoky autumn oud"
    )
    extra_words = set(re.findall(r"\b\w+\b", extra_query.lower())) if extra_query else set()

    top_n = st.slider("Number of recommendations", 5, 30, 10)

    # ── Score Parfumo dataset ──────────────────────────────────────────────────
    st.info("Scoring the Parfumo database against your taste profile…")

    # Combine taste profile words + any extra query words
    profile_words = set(top_notes) | set(top_themes) | extra_words

    discover_scores = []
    discover_infos  = []

    for _, row in parfumo.iterrows():

        # Skip fragrances the user already owns
        if str(row["Name"]).strip().lower() in owned_names:
            discover_scores.append(-1)
            discover_infos.append({})
            continue

        notes_words  = tokenize(row["All_Notes"])
        accord_words = tokenize(row["Main_Accords"])

        score = 0
        note_hits   = []
        accord_hits = []

        for word in profile_words:
            if word in notes_words:
                score += 4
                note_hits.append(word)
            if word in accord_words:
                score += 2
                accord_hits.append(word)

        # Weight by Parfumo community rating if available
        p_rating = row["Rating_Value"]
        if pd.notna(p_rating) and float(p_rating) > 0:
            score = score * (float(p_rating) / 10)

        discover_scores.append(score)
        discover_infos.append({
            "notes":  note_hits,
            "themes": accord_hits,
            "season": [], "strength": [], "gender": []
        })

    parfumo_copy = parfumo.copy()
    parfumo_copy["score"]      = discover_scores
    parfumo_copy["match_info"] = discover_infos

    results = parfumo_copy[parfumo_copy["score"] > 0].sort_values(
        "score", ascending=False
    ).head(top_n)

    if results.empty:
        st.warning("No results found. Try broadening your collection ratings.")
    else:
        st.subheader(f"Top {top_n} Recommendations for You")
        for _, row in results.iterrows():

            p_rating = row["Rating_Value"]
            rating_display = (
                f"{float(p_rating):.1f}/10 (Parfumo community)"
                if pd.notna(p_rating) else "Not rated"
            )

            render_match_card(
                row,
                row["match_info"],
                rating_col="Rating_Value",
                name_col="Name",
                brand_col="Brand",
                score_col="score",
                show_raw=False,
                extra_fields={
                    "Community Rating": rating_display,
                    "Main Accords":     str(row["Main_Accords"]),
                    "Notes":            str(row["All_Notes"])[:200] + "…"
                                        if len(str(row["All_Notes"])) > 200
                                        else str(row["All_Notes"]),
                    "URL": row["URL"]
                }
            )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":

    st.title("Collection Analytics")

    liked = collection[
        collection["Would I Wear?"].str.strip().str.lower() == "yes"
    ]

    # ── Top-line metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fragrances",   len(collection))
    m2.metric("Would Wear",         len(liked))
    m3.metric("Average Rating",     f"{collection['Rating'].mean():.2f}")
    m4.metric("5-Star Fragrances",  int((collection["Rating"] == 5).sum()))

    st.markdown("---")

    left, right = st.columns(2)

    # ── Rating distribution ────────────────────────────────────────────────────
    with left:
        st.subheader("Rating Distribution")
        rating_counts = collection["Rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)

    # ── Season breakdown ───────────────────────────────────────────────────────
    with right:
        st.subheader("Season Breakdown")
        season_counts = collection["Season"].value_counts()
        st.bar_chart(season_counts)

    left2, right2 = st.columns(2)

    # ── Strength breakdown ─────────────────────────────────────────────────────
    with left2:
        st.subheader("Strength Breakdown")
        strength_counts = collection["Strength"].value_counts()
        st.bar_chart(strength_counts)

    # ── Gender breakdown ───────────────────────────────────────────────────────
    with right2:
        st.subheader("Gender Breakdown")
        gender_counts = collection["Gender"].value_counts()
        st.bar_chart(gender_counts)

    st.markdown("---")

    # ── Favourite brands ───────────────────────────────────────────────────────
    st.subheader("Brands in Your Collection")
    brand_counts = collection["Brand"].value_counts().head(20)
    st.bar_chart(brand_counts)

    st.markdown("---")

    left3, right3 = st.columns(2)

    # ── Note frequency ─────────────────────────────────────────────────────────
    with left3:
        st.subheader("Top 20 Notes (liked fragrances)")
        note_counter = Counter()
        for _, row in liked.iterrows():
            for word in tokenize(row["Notes"]):
                if len(word) > 2:
                    note_counter[word] += 1
        top_notes_df = pd.DataFrame(
            note_counter.most_common(20), columns=["Note", "Count"]
        ).set_index("Note")
        st.bar_chart(top_notes_df)

    # ── Theme frequency ────────────────────────────────────────────────────────
    with right3:
        st.subheader("Top Themes (liked fragrances)")
        theme_counter = Counter()
        for _, row in liked.iterrows():
            for word in tokenize(row["Themes"]):
                if len(word) > 2:
                    theme_counter[word] += 1
        top_themes_df = pd.DataFrame(
            theme_counter.most_common(20), columns=["Theme", "Count"]
        ).set_index("Theme")
        st.bar_chart(top_themes_df)

    st.markdown("---")

    # ── Full collection table ──────────────────────────────────────────────────
    st.subheader("Full Collection")
    st.dataframe(
        collection[[
            "Name", "Brand", "Rating", "Would I Wear?",
            "Season", "Strength", "Gender", "Themes", "Notes"
        ]],
    )
