import streamlit as st
import pandas as pd
import re

## PAGE CONFIG
st.set_page_config(layout="wide")
page = st.sidebar.radio(
    "Navigation",
    [
        "Wear Today",
        "Discover New Fragrances",
        "Analytics"
    ]
)
collection = pd.read_csv(
        "PerfumeCollection.csv"
    )

## WEAR TODAY PAGE
if page == "Wear Today":

    st.title("Fragrance Recommendation Tool")

    st.write(f"You have {len(collection)} fragrances in your collection")

    col1, col2, col3 = st.columns(3)

    with col1:
        season_filter = st.selectbox(
            "Season",
            [
                "Any",
                "Spring",
                "Summer",
                "Autumn",
                "Winter"
            ]
        )

    with col2:
        strength_filter = st.selectbox(
            "Strength",
            [
                "Any",
                "Light",
                "Moderate",
                "Strong"
            ]
        )

    with col3:
        gender_filter = st.selectbox(
            "Gender",
            [
                "Any",
                "Masculine",
                "Feminine",
                "Unisex"
            ]
        )

    query = st.text_input(
        "Describe what you're looking for today",
        placeholder="cosy tea woody autumn"
    )

    if query:

        query_words = query.lower().split()

        scores = []
        match_infos = []

        for _, row in collection.iterrows():

            wear_flag = str(
                row["Would I Wear?"]
            ).strip().lower()

            if (
                season_filter != "Any"
                and season_filter.lower()
                not in str(row["Season"]).lower()
            ):
                scores.append(-1)
                match_infos.append({})
                continue

            if (
                strength_filter != "Any"
                and strength_filter.lower()
                not in str(row["Strength"]).lower()
            ):
                scores.append(-1)
                match_infos.append({})
                continue

            if (
                gender_filter != "Any"
                and gender_filter.lower()
                not in str(row["Gender"]).lower()
            ):
                scores.append(-1)
                match_infos.append({})
                continue

            if wear_flag == "no":
                scores.append(-1)
                match_infos.append({})
                continue

            theme_text = str(
                row["Themes"]
            ).lower()

            notes_text = str(
                row["Notes"]
            ).lower()

            season_text = str(
                row["Season"]
            ).lower()

            strength_text = str(
                row["Strength"]
            ).lower()

            gender_text = str(
                row["Gender"]
            ).lower()

            theme_words = set(
                re.findall(r"\b\w+\b", theme_text)
            )

            notes_words = set(
                re.findall(r"\b\w+\b", notes_text)
            )

            season_words = set(
                re.findall(r"\b\w+\b", season_text)
            )

            strength_words = set(
                re.findall(r"\b\w+\b", strength_text)
            )

            gender_words = set(
                re.findall(r"\b\w+\b", gender_text)
            )

            score = 0

            theme_matches = []
            note_matches = []
            season_matches = []
            strength_matches = []
            gender_matches = []

            for word in query_words:

                if word in notes_words:
                    score += 4
                    note_matches.append(word)

                if word in theme_words:
                    score += 2
                    theme_matches.append(word)

                if word in season_words:
                    score += 1
                    season_matches.append(word)

                if word in strength_words:
                    score += 1
                    strength_matches.append(word)

                if word in gender_words:
                    score += 1
                    gender_matches.append(word)

            rating = float(row["Rating"])

            score *= (rating / 5)

            scores.append(score)

            match_infos.append({
                "notes": note_matches,
                "themes": theme_matches,
                "season": season_matches,
                "strength": strength_matches,
                "gender": gender_matches
            })

        collection["score"] = scores
        collection["match_info"] = match_infos

        results = collection.sort_values(
            "score",
            ascending=False
        )

        results = results[
            results["score"] > 0
        ].head(10)

        st.subheader("Recommended")

        for _, row in results.iterrows():

            rating = row["Rating"]

            if rating == 5:
                st.markdown(
                    f"""
                    <h2 style='color:#2E8B57'>
                    {row['Name']}
                    </h2>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"## {row['Name']}"
                )

            st.write(
                f"**Brand:** {row['Brand']}"
            )

            st.write(
                f"**Rating:** {rating}/5"
            )

            st.write(
                f"**Match Score:** {round(row['score'], 1)}"
            )

            info = row["match_info"]

            reasons = []

            if info["notes"]:
                reasons.append(
                    f"contains notes: {', '.join(info['notes'])}"
                )

            if info["themes"]:
                reasons.append(
                    f"matches themes: {', '.join(info['themes'])}"
                )

            if info["season"]:
                reasons.append(
                    f"suitable for {', '.join(info['season'])}"
                )

            if float(rating) >= 5:
                reasons.append(
                    "one of your 5-star fragrances"
                )
            
            if reasons:

                st.success(
                    "Why it matched: "
                    + "; ".join(reasons)
                )

            if info["notes"]:
                st.write(
                    f"📝 Notes matched: {', '.join(info['notes'])}"
                )

            if info["themes"]:
                st.write(
                    f"🎭 Themes matched: {', '.join(info['themes'])}"
                )

            if info["season"]:
                st.write(
                    f"🍂 Season matched: {', '.join(info['season'])}"
                )

            if info["strength"]:
                st.write(
                    f"💪 Strength matched: {', '.join(info['strength'])}"
                )

            if info["gender"]:
                st.write(
                    f"👤 Gender matched: {', '.join(info['gender'])}"
                )

            st.write(
                f"**Themes:** {row['Themes']}"
            )

            st.write(
                f"**Notes:** {row['Notes']}"
            )

            st.markdown("---")

## DISCOVER NEW PERFUMES PAGE
elif page == "Discover New Perfumes":

    st.header(
        "Discover New Perfumes"
    )

    st.write(
        "Coming next..."
    )

## ANALYTICS PAGE
elif page == "Analytics":

    st.header(
        "Collection Analytics"
    )

    st.write(
        f"Collection Size: {len(collection)}"
    )

    st.write(
        f"Average Rating: {collection['Rating'].mean():.2f}"
    )