import os
import math
import re
import random
import shutil
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fragrance Recommender")

# ── PASSWORD GATE ──────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Fragrance Recommender")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pwd == st.secrets.get("app_password", ""):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

password = st.sidebar.text_input("Password", type="password")
if password != st.secrets.get("app_password", ""):
    st.warning("Enter the password to access the app.")
    st.stop()

# ── MOBILE-FRIENDLY CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .block-container {
        padding: 1rem 0.75rem !important;
        max-width: 100% !important;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    .stButton > button {
        width: 100% !important;
        margin-bottom: 0.4rem !important;
        min-height: 2.5rem !important;
    }
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select {
        font-size: 16px !important;
    }
    [data-testid="metric-container"] { padding: 0.5rem !important; }
    .stAlert { padding: 0.5rem 0.75rem !important; }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── MOBILE-FRIENDLY CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@media (max-width: 768px) {
    /* Stack all column layouts vertically */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* Tighter page padding */
    .block-container {
        padding: 1rem 0.75rem !important;
        max-width: 100% !important;
    }
    /* Smaller headings */
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    /* Full-width buttons, easier to tap */
    .stButton > button {
        width: 100% !important;
        margin-bottom: 0.4rem !important;
        min-height: 2.5rem !important;
    }
    /* Prevent iOS font size zoom on input focus */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select {
        font-size: 16px !important;
    }
    /* Tighter metric cards */
    [data-testid="metric-container"] {
        padding: 0.5rem !important;
    }
    /* Tighter alerts */
    .stAlert { padding: 0.5rem 0.75rem !important; }
    /* Make tabs scrollable if too many */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
}
</style>
""", unsafe_allow_html=True)

def responsive_columns(n_desktop, gap="small"):
    """
    Wrapper around st.columns. Accepts int or list (like st.columns).
    The CSS media query stacks columns on narrow viewports automatically.
    """
    try:
        return st.columns(n_desktop, gap=gap)
    except TypeError:
        # Older Streamlit versions don't support gap parameter
        return st.columns(n_desktop)

page = st.sidebar.radio(
    "Navigation",
    ["My Collection", "Wear Today", "Discover New Fragrances", "Analytics"]
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
NOTE_STOPWORDS = {
    "notes", "woody", "and", "the", "with", "of", "a", "an",
    "von", "de", "du", "la", "le"
}
MODEL_NAME            = "all-MiniLM-L6-v2"
PARFUMO_CACHE_PATH    = "parfumo_embeddings.npy"
COLLECTION_CACHE_PATH = "collection_embeddings.npy"
REFINE_BOOST          = 20
SEMANTIC_WEIGHT       = 0.6   # blend ratio for Wear Today (semantic vs keyword)

WISHLIST_PATH = "Wishlist.csv"  # kept for local fallback only

# ── GOOGLE SHEETS CLIENT ───────────────────────────────────────────────────────
@st.cache_resource
def get_gsheet_client():
    """Authorise and return a gspread client using Streamlit secrets."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    return gspread.authorize(creds)

def get_worksheet(sheet_name):
    """Return a gspread worksheet by tab name."""
    client = get_gsheet_client()
    spreadsheet_id = st.secrets["sheets"]["spreadsheet_id"]
    sh = client.open_by_key(spreadsheet_id)
    return sh.worksheet(sheet_name)

def read_sheet(sheet_name):
    """Read a Google Sheet tab and return a pandas DataFrame."""
    ws = get_worksheet(sheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)

def write_sheet(sheet_name, df):
    """Overwrite a Google Sheet tab with a pandas DataFrame."""
    ws = get_worksheet(sheet_name)
    ws.clear()
    ws.update([df.columns.tolist()] + df.fillna("").astype(str).values.tolist())

def invalidate_data_cache():
    """Clear Streamlit's data cache so next load reads fresh from Sheets."""
    st.cache_data.clear()

# ── HELPERS ────────────────────────────────────────────────────────────────────
def backup_collection():
    """
    Back up the Collection sheet to a CollectionBackup tab in Google Sheets.
    Falls back to local file backup if Sheets unavailable.
    """
    try:
        df = read_sheet("Collection")
        write_sheet("CollectionBackup", df)
    except Exception:
        if os.path.exists("PerfumeCollection.csv"):
            shutil.copy2("PerfumeCollection.csv", "PerfumeCollection_backup.csv")
def tokenize(text):
    return set(re.findall(r"\b\w+\b", str(text).lower()))

def make_doc(row, notes_col="Notes", themes_col="Themes"):
    """Build a plain-text description for embedding."""
    notes  = str(row.get(notes_col,  "")).strip()
    themes = str(row.get(themes_col, "")).strip()
    return f"{themes}. {notes}".strip(". ")

def make_parfumo_doc(row):
    notes   = str(row["All_Notes"]).strip()
    accords = str(row["Main_Accords"]).strip()
    return f"{accords}. {notes}".strip(". ")

def scrape_parfumo_url(url):
    import json as _json
    from bs4 import BeautifulSoup
    import requests

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return None, f"Could not fetch page: {e}"

    try:
        soup = BeautifulSoup(r.text, "html.parser")

        # ── Name + Brand ──────────────────────────────────────────────────────
        name = ""
        brand = ""
        title_el = soup.find("title")
        if title_el:
            t = title_el.get_text(strip=True)
            for sep in [" by ", " - "]:
                if sep in t:
                    parts = t.split(sep)
                    name = parts[0].strip()
                    brand_raw = parts[1]
                    # Strip trailing noise: " » Reviews...", "| Parfumo", "- Parfumo" etc.
                    brand_raw = brand_raw.split(" » ")[0]
                    brand_raw = brand_raw.split(" | ")[0]
                    brand_raw = brand_raw.split(" - ")[0]
                    # Strip concentration suffixes like "(Eau de Parfum)", "(EDT)" etc.
                    brand_raw = re.sub(r'\s*\(.*?\)\s*$', '', brand_raw)
                    brand = brand_raw.strip()
                    break
        if not name:
            path_parts = [p for p in url.rstrip("/").split("/") if p]
            if len(path_parts) >= 2:
                name  = path_parts[-1].replace("_", " ").replace("-", " ")
                brand = path_parts[-2].replace("_", " ").replace("-", " ")

        # ── Themes / Main Accords ─────────────────────────────────────────────
        # Strategy 1: look near a "Main Accords" text anchor
        seen, accords_out = set(), []
        anchor = soup.find(string=lambda s: s and "main accord" in s.lower())
        if anchor:
            section = anchor.find_parent()
            for _ in range(4):  # walk up a few levels to find the container
                if section and section.parent:
                    section = section.parent
                    candidates = section.find_all(["span", "a", "div", "li"])
                    for el in candidates:
                        cls = " ".join(el.get("class", []))
                        t = el.get_text(strip=True)
                        if 2 < len(t) < 50 and t.lower() not in seen and "accord" not in t.lower():
                            seen.add(t.lower())
                            accords_out.append(t)
                    if accords_out:
                        break

        # Strategy 2: any element with "accord" in its class
        if not accords_out:
            for el in soup.find_all(True):
                cls = " ".join(el.get("class", []))
                if "accord" in cls.lower():
                    t = el.get_text(strip=True)
                    if 2 < len(t) < 50 and t.lower() not in seen:
                        seen.add(t.lower())
                        accords_out.append(t)

        # Strategy 3: look for JSON-LD structured data
        if not accords_out:
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = _json.loads(script.string or "")
                    for key in ("keywords", "description"):
                        if key in data and isinstance(data[key], str):
                            accords_out = [w.strip() for w in data[key].split(",") if w.strip()]
                            break
                except Exception:
                    pass
                if accords_out:
                    break

        # Remove entries that are concatenations of other entries (parent container text)
        accords_out = [a for a in accords_out if not any(
            a != b and b.lower() in a.lower() and len(a) > len(b)
            for b in accords_out
        )]
        themes = ", ".join(accords_out[:12])

        # ── Notes ─────────────────────────────────────────────────────────────
        seen, notes_out = set(), []

        # Words that indicate we've grabbed page chrome rather than a note name
        NOTES_JUNK = {
            "smell", "feel", "compare", "more", "pronunciation", "reviews",
            "ratings", "rated", "ranking", "ranked", "men", "women", "unisex",
            "parfumo", "lvmh", "verified", "source", "backed", "rating",
            "collection", "brand", "house", "perfumer", "year", "ml",
            "opomone", "givaudan", "firmenich", "iff", "symrise", "takasago",
            "vegan", "natural", "organic", "cruelty", "free", "niche",
            "exclusive", "limited", "edition", "new", "vintage",
        }

        def is_valid_note(text):
            t = text.strip()
            if not (2 < len(t) < 50):
                return False
            cleaned = re.sub(r'\b(CO2?|absolute|extract|accord)\b', '', t, flags=re.IGNORECASE).strip()
            if any(ch.isdigit() for ch in cleaned):
                return False
            if re.search(r'[^\w\s\-\']', t):
                return False
            words = set(re.findall(r'\b\w+\b', t.lower()))
            if words & NOTES_JUNK:
                return False
            # Block CamelCase concatenations like "FreshCitrusAquaticGreenSpicy"
            if re.search(r'[a-z][A-Z]', t) and len(t) > 15:
                return False
            return True

        # Strategy 1a: look near pyramid labels (Top / Heart / Middle / Base)
        for label in ["top notes", "heart notes", "middle notes", "base notes"]:
            anchor = soup.find(string=lambda s: s and label in s.lower())
            if anchor:
                section = anchor.find_parent()
                for _ in range(3):
                    if section and section.parent:
                        section = section.parent
                        candidates = section.find_all(["span", "a", "li"])
                        batch = []
                        for el in candidates:
                            if el.name == "a" and "/Users/" in (el.get("href") or ""):
                                continue
                            t = el.get_text(strip=True)
                            if is_valid_note(t) and t.lower() not in seen:
                                seen.add(t.lower())
                                batch.append(t)
                        if batch:
                            notes_out.extend(batch)
                            break

        # Strategy 1b: "Fragrance Notes" single-section format (e.g. LV Imagination)
        if not notes_out:
            for label in ["fragrance notes", "notes"]:
                anchor = soup.find(string=lambda s: s and s.strip().lower() == label)
                if anchor:
                    section = anchor.find_parent()
                    for _ in range(4):
                        if section and section.parent:
                            section = section.parent
                            candidates = section.find_all(["span", "a", "li"])
                            batch = []
                            for el in candidates:
                                if el.name == "a" and "/Users/" in (el.get("href") or ""):
                                    continue
                                t = el.get_text(strip=True)
                                if is_valid_note(t) and t.lower() not in seen:
                                    seen.add(t.lower())
                                    batch.append(t)
                            if batch:
                                notes_out.extend(batch)
                                break
                    if notes_out:
                        break

        # Strategy 2: elements with "note" in their CSS class (fallback)
        if not notes_out:
            for el in soup.find_all(True):
                cls = " ".join(el.get("class", []))
                if "note" in cls.lower():
                    if el.name == "a" and "/Users/" in (el.get("href") or ""):
                        continue
                    t = el.get_text(strip=True)
                    if is_valid_note(t) and t.lower() not in seen:
                        seen.add(t.lower())
                        notes_out.append(t)

        notes = ", ".join(notes_out[:40])

        # ── Post-process: remove brand name and perfumer-like entries from notes
        # Perfumer names are typically two capitalised words with no geographic
        # prefix — filter them by checking they don't contain a known origin word
        GEOGRAPHIC_PREFIXES = {
            "calabrian", "nigerian", "ceylonese", "sicilian", "tunisian",
            "french", "moroccan", "indian", "bulgarian", "turkish", "egyptian",
            "chinese", "japanese", "brazilian", "haitian", "indonesian",
            "sri", "south", "north", "west", "east", "black", "white", "pink",
            "green", "red", "blue", "dark", "light", "wild", "royal",
        }
        def looks_like_perfumer(text, brand_name):
            t = text.strip()
            # Remove if it's the brand name
            if t.lower() == brand_name.lower():
                return True
            words = t.split()
            # Single-word entries that are CamelCase or mixed-internal-caps
            # (e.g. OPomone, GivaudanHouse) are likely organisation names, not notes
            if len(words) == 1 and re.search(r'[A-Z][a-z]+[A-Z]|[a-z][A-Z]', t):
                return True
            # Perfumer names: 2+ words, all Title Case, no geographic prefix
            if (len(words) >= 2
                    and all(w[0].isupper() for w in words if w)
                    and not any(w.lower() in GEOGRAPHIC_PREFIXES for w in words)):
                if all(w[0].isupper() for w in words):
                    return True
            return False

        clean_notes = [
            n for n in notes_out
            if not looks_like_perfumer(n, brand)
        ]
        notes = ", ".join(clean_notes[:40])

        return {"Name": name, "Brand": brand, "Themes": themes, "Notes": notes, "URL": url}, None

    except Exception as e:
        return None, f"Could not parse page: {e}"

# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_collection(version=0):
    """Load collection from Google Sheets. TTL of 30s so edits propagate quickly."""
    try:
        df = read_sheet("Collection")
        # Ensure Rating is numeric
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception:
        # Fallback to local CSV if Sheets unavailable (e.g. local dev without secrets)
        return pd.read_csv("PerfumeCollection.csv")

@st.cache_data
def load_parfumo():
    df = pd.read_csv("parfumo_data_clean.csv")
    def combine_notes(row):
        parts = [str(row["Top_Notes"]), str(row["Middle_Notes"]), str(row["Base_Notes"])]
        return ", ".join(p for p in parts if p.lower() != "nan")
    df["All_Notes"] = df.apply(combine_notes, axis=1)
    return df

@st.cache_data
def compute_idf(parfumo):
    doc_count  = Counter()
    total_docs = 0
    for _, row in parfumo.iterrows():
        words = tokenize(row["All_Notes"]) | tokenize(str(row["Main_Accords"]))
        words -= NOTE_STOPWORDS
        if words:
            total_docs += 1
            for w in words:
                doc_count[w] += 1
    return {w: math.log(total_docs / (1 + c)) for w, c in doc_count.items()}

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def get_collection_embeddings(collection, _model):
    """Embed each collection fragrance. Cached to disk."""
    if os.path.exists(COLLECTION_CACHE_PATH):
        return np.load(COLLECTION_CACHE_PATH)
    docs = [make_doc(row) for _, row in collection.iterrows()]
    embs = _model.encode(docs, show_progress_bar=False)
    np.save(COLLECTION_CACHE_PATH, embs)
    return embs

@st.cache_data
def get_parfumo_embeddings(parfumo, _model):
    """Embed all Parfumo fragrances. Cached to disk (~few minutes first run)."""
    if os.path.exists(PARFUMO_CACHE_PATH):
        return np.load(PARFUMO_CACHE_PATH)
    docs = [make_parfumo_doc(row) for _, row in parfumo.iterrows()]
    with st.spinner("Computing Parfumo embeddings for the first time — this takes a few minutes but only happens once…"):
        embs = _model.encode(docs, batch_size=256, show_progress_bar=False)
    np.save(PARFUMO_CACHE_PATH, embs)
    return embs

@st.cache_data(ttl=30)
def load_wishlist(version=0):
    """Load wishlist from Google Sheets."""
    try:
        df = read_sheet("Wishlist")
        if df.empty:
            return pd.DataFrame(columns=["Name", "Brand", "Notes", "Themes", "URL"])
        df = df.drop(columns=["Rating_Value"], errors="ignore")
        if "Main_Accords" in df.columns and "Themes" not in df.columns:
            df = df.rename(columns={"Main_Accords": "Themes"})
        return df
    except Exception:
        # Fallback to local CSV
        if not os.path.exists(WISHLIST_PATH):
            return pd.DataFrame(columns=["Name", "Brand", "Notes", "Themes", "URL"])
        df = pd.read_csv(WISHLIST_PATH)
        df = df.drop(columns=["Rating_Value"], errors="ignore")
        if "Main_Accords" in df.columns and "Themes" not in df.columns:
            df = df.rename(columns={"Main_Accords": "Themes"})
        return df

# ── PROFILE BUILDERS ───────────────────────────────────────────────────────────
def build_taste_profile(collection, idf):
    liked = collection[collection["Would I Wear?"].str.strip().str.lower() == "yes"]
    note_counter  = Counter()
    theme_counter = Counter()
    for _, row in liked.iterrows():
        rw = float(row["Rating"]) / 5.0
        for word in tokenize(row["Notes"]):
            if len(word) > 2 and word not in NOTE_STOPWORDS:
                note_counter[word] += rw * idf.get(word, 1.0)
        for word in tokenize(row["Themes"]):
            if len(word) > 2 and word not in NOTE_STOPWORDS:
                theme_counter[word] += rw
    return note_counter, theme_counter

def build_brand_profile(collection):
    liked = collection[collection["Would I Wear?"].str.strip().str.lower() == "yes"]
    brand_counter = Counter()
    for _, row in liked.iterrows():
        brand_counter[str(row["Brand"]).strip().lower()] += float(row["Rating"]) / 5.0
    max_score = max(brand_counter.values()) if brand_counter else 1
    return {b: s / max_score for b, s in brand_counter.items()}

# ── KEYWORD SCORER ─────────────────────────────────────────────────────────────
def score_row_against_words(query_words, notes_words, theme_words,
                             season_words, strength_words, gender_words):
    score = 0
    note_matches = []; theme_matches = []; season_matches = []
    strength_matches = []; gender_matches = []
    for word in query_words:
        if word in notes_words:    score += 4; note_matches.append(word)
        if word in theme_words:    score += 2; theme_matches.append(word)
        if word in season_words:   score += 1; season_matches.append(word)
        if word in strength_words: score += 1; strength_matches.append(word)
        if word in gender_words:   score += 1; gender_matches.append(word)
    return score, {
        "notes": note_matches, "themes": theme_matches,
        "season": season_matches, "strength": strength_matches,
        "gender": gender_matches
    }

# ── CARD RENDERER ──────────────────────────────────────────────────────────────
def render_match_card(row, match_info, rating_col="Rating",
                      name_col="Name", brand_col="Brand",
                      score_col="score", show_raw=True,
                      extra_fields=None, wishlist_key=None):
    rating = row[rating_col]
    name   = row[name_col]
    brand  = row[brand_col]
    score  = row[score_col]

    if float(rating) == 5:
        st.markdown(f"<h2 style='color:#2E8B57'>{name}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"## {name}")

    st.write(f"**Brand:** {brand}")
    st.write(f"**Rating:** {rating}")
    st.write(f"**Match Score:** {round(float(score), 1)}")

    if extra_fields:
        for label, value in extra_fields.items():
            st.write(f"**{label}:** {value}")

    reasons = []
    if match_info.get("semantic_query"):
        reasons.append(f"semantically matches \"{match_info['semantic_query']}\"")
    if match_info.get("refine_hits"):
        reasons.append(f"matches your search: {', '.join(match_info['refine_hits'])}")
    if match_info.get("semantic_refine"):
        reasons.append(f"semantically matches \"{match_info['semantic_refine']}\"")
    if match_info.get("notes"):
        reasons.append(f"contains notes: {', '.join(match_info['notes'])}")
    if match_info.get("themes"):
        reasons.append(f"matches themes: {', '.join(match_info['themes'])}")
    if match_info.get("season"):
        reasons.append(f"suitable for {', '.join(match_info['season'])}")
    if match_info.get("brand_match"):
        reasons.append(f"from a brand you already love ({brand})")
    if match_info.get("is_collection") and float(rating) >= 5:
        reasons.append("one of your 5-star fragrances")

    if reasons:
        st.success("Why it matched: " + "; ".join(reasons))

    if show_raw:
        with st.expander("Notes & Themes"):
            if "Themes" in row.index:
                st.write(f"**Themes:** {row['Themes']}")
            if "Notes" in row.index:
                st.write(f"**Notes:** {row['Notes']}")

    if wishlist_key is not None:
        if row[name_col].strip().lower() in wishlist["Name"].str.strip().str.lower().tolist():
            st.caption("✓ Already on your wishlist")
        elif st.button("+ Add to Wishlist", key=wishlist_key):
            new_wish = pd.DataFrame([{
                "Name":   row[name_col],
                "Brand":  row[brand_col],
                "Notes":  row.get("All_Notes", ""),
                "Themes": row.get("Main_Accords", ""),
                "URL":    row.get("URL", ""),
            }])
            updated_wish = pd.concat([wishlist, new_wish], ignore_index=True)
            write_sheet("Wishlist", updated_wish)
            st.session_state.wishlist_version += 1
            invalidate_data_cache()
            st.toast(f"'{row[name_col]}' added to your wishlist!")
            st.rerun()

    st.markdown("---")


# ── LOAD EVERYTHING ────────────────────────────────────────────────────────────
if "last_add_success" not in st.session_state:
    st.session_state.last_add_success = None
if "collection_version" not in st.session_state:
    st.session_state.collection_version = 0
if "add_form_key" not in st.session_state:
    st.session_state.add_form_key = 0
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None
if "last_surprise" not in st.session_state:
    st.session_state.last_surprise = None
if "wishlist_version" not in st.session_state:
    st.session_state.wishlist_version = 0
if "wishlist_remove_prompt" not in st.session_state:
    st.session_state.wishlist_remove_prompt = None
if "discover_find_active" not in st.session_state:
    st.session_state.discover_find_active = False
if "manual_wish_form_key" not in st.session_state:
    st.session_state.manual_wish_form_key = 0
if "collection_upload_key" not in st.session_state:
    st.session_state.collection_upload_key = 0
if "wishlist_upload_key" not in st.session_state:
    st.session_state.wishlist_upload_key = 0
if "clear_add_form_pending" not in st.session_state:
    st.session_state.clear_add_form_pending = False
if "scraped_parfumo" not in st.session_state:
    st.session_state.scraped_parfumo = None
if "scraped_parfumo_form_key" not in st.session_state:
    st.session_state.scraped_parfumo_form_key = 0
if "parfumo_url_key" not in st.session_state:
    st.session_state.parfumo_url_key = 0
if "add_parfumo_url_key" not in st.session_state:
    st.session_state.add_parfumo_url_key = 0

collection = load_collection(st.session_state.collection_version).copy()
wishlist   = load_wishlist(st.session_state.wishlist_version).copy()
parfumo    = load_parfumo()
idf        = compute_idf(parfumo)
model      = load_model()
col_embs   = get_collection_embeddings(collection, model)


# ══════════════════════════════════════════════════════════════════════════════
# WEAR TODAY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Wear Today":

    st.title("Fragrance Recommendation Tool")
    st.write(f"You have **{len(collection)}** fragrances in your collection")

    # ── Surprise Me ───────────────────────────────────────────────────────────
    # Filter to fragrances you'd wear and rated 4 or 5 stars.
    # random.choice() picks one at random from that list.
    # We exclude the last pick (stored in session_state) so you don't get
    # the same fragrance twice in a row.
    surprise_pool = collection[
        (collection["Would I Wear?"].str.strip().str.lower() == "yes") &
        (collection["Rating"] >= 4)
    ]
    if st.button("Surprise Me"):
        # Build a pool that excludes the last pick if possible
        candidates = surprise_pool[
            surprise_pool["Name"] != st.session_state.last_surprise
        ]
        # If excluding last pick leaves nothing (very small collection), use full pool
        if candidates.empty:
            candidates = surprise_pool
        if not candidates.empty:
            pick = candidates.sample(1).iloc[0]
            st.session_state.last_surprise = pick["Name"]
            stars = "★" * int(pick["Rating"]) + "☆" * (5 - int(pick["Rating"]))
            st.success(f"**Today, wear: {pick['Name']}** by {pick['Brand']}")
            st.write(f"**Rating:** {stars}")
            st.write(f"**Notes:** {pick['Notes']}")
            st.write(f"**Themes:** {pick['Themes']}")
            st.write(f"**Season:** {pick['Season']}  |  **Strength:** {pick['Strength']}")
        else:
            st.warning("No 4+ star wearable fragrances found in your collection.")

    st.markdown("---")
    semantic_query = st.text_input(
        "Describe what you're looking for today",
        placeholder="something cosy and dark for a rainy evening"
    )
    keyword_query = st.text_input(
        "Match specific notes or themes (optional)",
        placeholder="tea vanilla petrichor"
    )

    col1, col2, col3 = responsive_columns(3)
    with col1:
        season_filter = st.selectbox("Season",   ["Any","Spring","Summer","Autumn","Winter"])
    with col2:
        strength_filter = st.selectbox("Strength", ["Any","Normal", "Subtle", "Strong"])
    with col3:
        gender_filter = st.selectbox("Gender",   ["Any","Masculine","Feminine","Unisex"])

    keyword_words = set(re.findall(r"\b\w+\b", keyword_query.lower())) if keyword_query else set()

    search_clicked = st.button("Find Fragrances")

    if not search_clicked and not semantic_query and not keyword_words:
        st.info("Use the filters and search boxes above, then click **Find Fragrances**.")
    elif search_clicked or semantic_query or keyword_words:
        # ── Semantic scores for whole collection ──────────────────────────────
        sem_scores = None
        if semantic_query:
            query_emb  = model.encode([semantic_query])
            sim        = cosine_similarity(query_emb, col_embs)[0]
            sem_scores = sim

        scores      = []
        match_infos = []

        for i, (_, row) in enumerate(collection.iterrows()):

            wear_flag = str(row["Would I Wear?"]).strip().lower()

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

            # ── No query: rank by rating ───────────────────────────────────────
            if not semantic_query and not keyword_words:
                scores.append(float(row["Rating"]))
                match_infos.append({
                    "notes": [], "themes": [], "season": [],
                    "strength": [], "gender": [], "is_collection": True
                })
                continue

            notes_words    = tokenize(row["Notes"])
            theme_words    = tokenize(row["Themes"])
            season_words   = tokenize(row["Season"])
            strength_words = tokenize(row["Strength"])
            gender_words   = tokenize(row["Gender"])
            rating         = float(row["Rating"])

            # ── Keyword score ──────────────────────────────────────────────────
            kw_score, info = score_row_against_words(
                keyword_words, notes_words, theme_words,
                season_words, strength_words, gender_words
            )
            kw_score = kw_score * (rating / 5)

            # ── Semantic score ─────────────────────────────────────────────────
            sem_score = 0.0
            if sem_scores is not None:
                sem_score = float(sem_scores[i]) * 20 * (rating / 5)

            # ── Blend ─────────────────────────────────────────────────────────
            if semantic_query and keyword_words:
                final_score = (SEMANTIC_WEIGHT * sem_score) + ((1 - SEMANTIC_WEIGHT) * kw_score)
            elif semantic_query:
                final_score = sem_score
            else:
                final_score = kw_score

            scores.append(final_score)
            match_infos.append({
                **info,
                "is_collection":  True,
                "semantic_query": semantic_query if semantic_query else None
            })

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
                render_match_card(row, row["match_info"], rating_col="Rating", show_raw=True)


# ══════════════════════════════════════════════════════════════════════════════
# DISCOVER NEW FRAGRANCES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Discover New Fragrances":

    st.title("Discover New Fragrances")

    st.write(
        "Based on what you love in your collection, "
        "here are fragrances from the Parfumo database you don't own yet."
    )

    note_profile, theme_profile = build_taste_profile(collection, idf)
    brand_profile = build_brand_profile(collection)

    top_notes  = [w for w, _ in note_profile.most_common(20)]
    top_themes = [w for w, _ in theme_profile.most_common(10)]
    top_brands = sorted(brand_profile.items(), key=lambda x: x[1], reverse=True)

    with st.expander("Your taste profile (learned from your collection)"):
        st.write(f"**Favourite notes:** {', '.join(top_notes)}")
        st.write(f"**Favourite themes / accords:** {', '.join(top_themes)}")
        st.write(f"**Favourite brands:** {', '.join(b for b, _ in top_brands[:10])}")

    owned_names    = set(collection["Name"].str.strip().str.lower())
    wishlist_names = set(wishlist["Name"].str.strip().str.lower())

    # ── Similar to This ───────────────────────────────────────────────────────
    # Let the user pick a fragrance they already own and find similar ones
    # in the Parfumo database using semantic similarity.
    st.subheader("Find fragrances similar to one you already love")
    similar_source = st.selectbox(
        "Pick a fragrance from your collection",
        ["(none)"] + sorted(collection["Name"].tolist()),
        key="similar_source"
    )

    if similar_source != "(none)":
        # Get the index of the chosen fragrance in the collection dataframe.
        # We need this to look up its pre-computed embedding from col_embs.
        source_idx = collection.reset_index(drop=True).index[
            collection.reset_index(drop=True)["Name"] == similar_source
        ][0]

        # Load Parfumo embeddings (cached to disk after first run)
        par_embs = get_parfumo_embeddings(parfumo, model)

        # col_embs[source_idx] is the embedding vector for the chosen fragrance.
        # We reshape it to (1, N) so cosine_similarity can compare it against
        # all rows of par_embs at once, returning one similarity score per row.
        source_emb    = col_embs[source_idx].reshape(1, -1)
        sim_scores    = cosine_similarity(source_emb, par_embs)[0]

        sim_top_n = st.slider("Number of similar fragrances to show", 5, 20, 10, key="sim_slider")

        # Build a dataframe of results, excluding fragrances already owned
        parfumo_copy2 = parfumo.copy()
        parfumo_copy2["sim_score"] = sim_scores
        parfumo_copy2 = parfumo_copy2[
            ~parfumo_copy2["Name"].str.strip().str.lower().isin(owned_names) &
            ~parfumo_copy2["Name"].str.strip().str.lower().isin(wishlist_names)
        ]
        sim_results = parfumo_copy2.sort_values("sim_score", ascending=False).head(sim_top_n)

        st.write(f"Fragrances most similar to **{similar_source}**:")
        with st.container(height=500):
            for _, row in sim_results.iterrows():
                p_rating = row["Rating_Value"]
                rating_display = f"{float(p_rating):.1f}/10" if pd.notna(p_rating) else "Not rated"
                st.markdown(f"### {row['Name']}")
                st.write(f"**Brand:** {row['Brand']}  |  **Community Rating:** {rating_display}")
                st.write(f"**Main Accords:** {row['Main_Accords']}")
                st.write(f"**Notes:** {str(row['All_Notes'])[:200]}")
                st.write(f"**Similarity:** {row['sim_score']:.2f}  |  **URL:** {row['URL']}")
                if row["Name"].strip().lower() in wishlist["Name"].str.strip().str.lower().tolist():
                    st.caption("✓ Already on your wishlist")
                elif st.button("+ Add to Wishlist", key=f"sim_wish_{row['Name']}"):
                    new_wish = pd.DataFrame([{
                        "Name":   row["Name"],
                        "Brand":  row["Brand"],
                        "Notes":  row["All_Notes"],
                        "Themes": row["Main_Accords"],
                        "URL":    row["URL"],
                    }])
                    updated_wish = pd.concat([wishlist, new_wish], ignore_index=True)
                    write_sheet("Wishlist", updated_wish)
                    st.session_state.wishlist_version += 1
                    invalidate_data_cache()
                    st.toast(f"'{row['Name']}' added to your wishlist!")
                    st.rerun()
                st.markdown("---")

    st.markdown("---")
    extra_query = st.text_input(
        "Refine by notes or themes (optional)",
        placeholder="e.g. smoky oud leather"
    )
    extra_words = set(re.findall(r"\b\w+\b", extra_query.lower())) if extra_query else set()

    semantic_refine = st.text_input(
        "Semantic search (optional) — describe a feeling or vibe",
        placeholder="e.g. dark gothic atmosphere like a Victorian library"
    )

    top_n            = st.slider("Number of recommendations", 5, 30, 10)
    min_rating_count = st.slider("Minimum community ratings (filters obscure fragrances)", 0, 500, 50)

    if st.button("Find Fragrances"):
        st.session_state.discover_find_active = True

    if not st.session_state.discover_find_active:
        st.info("Set your preferences above and click **Find Fragrances** to get recommendations.")
    else:
        # ── Pre-compute Parfumo embeddings if semantic refine is used ─────────
        par_embs = None
        sem_refine_scores = None
        if semantic_refine:
            par_embs = get_parfumo_embeddings(parfumo, model)
            refine_emb        = model.encode([semantic_refine])
            sem_refine_scores = cosine_similarity(refine_emb, par_embs)[0]

        st.info("Scoring the Parfumo database against your taste profile…")

        profile_words = set(top_notes) | set(top_themes)

        discover_scores = []
        discover_infos  = []

        for i, (_, row) in enumerate(parfumo.iterrows()):

            if str(row["Name"]).strip().lower() in owned_names:
                discover_scores.append(-1); discover_infos.append({}); continue

            if str(row["Name"]).strip().lower() in wishlist_names:
                discover_scores.append(-1); discover_infos.append({}); continue

            r_count = row["Rating_Count"]
            if pd.isna(r_count) or float(r_count) < min_rating_count:
                discover_scores.append(-1); discover_infos.append({}); continue

            notes_words  = tokenize(row["All_Notes"])
            accord_words = tokenize(row["Main_Accords"])

            # ── Profile keyword score ──────────────────────────────────────────
            profile_score = 0
            note_hits     = []
            accord_hits   = []
            for word in profile_words:
                word_idf = idf.get(word, 1.0)
                if word in notes_words:  profile_score += 4 * word_idf; note_hits.append(word)
                if word in accord_words: profile_score += 2 * word_idf; accord_hits.append(word)

            # ── Keyword refine score ───────────────────────────────────────────
            refine_score = 0
            refine_hits  = []
            for word in extra_words:
                word_idf = idf.get(word, 1.0)
                if word in notes_words or word in accord_words:
                    refine_score += REFINE_BOOST * word_idf
                    refine_hits.append(word)

            if extra_words and refine_score == 0:
                discover_scores.append(-1); discover_infos.append({}); continue

            # ── Semantic refine score ──────────────────────────────────────────
            sem_score = 0.0
            if sem_refine_scores is not None:
                sem_score = float(sem_refine_scores[i]) * REFINE_BOOST * 4

            # If semantic refine is active, require a minimum similarity threshold
            if semantic_refine and float(sem_refine_scores[i]) < 0.15:
                discover_scores.append(-1); discover_infos.append({}); continue

            score = profile_score + refine_score + sem_score

            p_rating = row["Rating_Value"]
            if pd.notna(p_rating) and float(p_rating) > 0:
                score = score * (float(p_rating) / 10)

            brand_key = str(row["Brand"]).strip().lower()
            affinity  = brand_profile.get(brand_key, 0)
            score     = score * (1 + affinity)

            discover_scores.append(score)
            discover_infos.append({
                "notes":           note_hits,
                "themes":          accord_hits,
                "refine_hits":     refine_hits,
                "semantic_refine": semantic_refine if semantic_refine else None,
                "season":          [],
                "strength":        [],
                "gender":          [],
                "brand_match":     affinity > 0
            })

        parfumo_copy = parfumo.copy()
        parfumo_copy["score"]      = discover_scores
        parfumo_copy["match_info"] = discover_infos

        results = parfumo_copy[parfumo_copy["score"] > 0].sort_values(
            "score", ascending=False
        ).head(top_n)

        if results.empty:
            st.warning("No results found. Try adjusting your search or lowering the minimum rating count.")
        else:
            st.subheader(f"Top {top_n} Recommendations for You")
            with st.container(height=600):
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
                            "Themes":          str(row["Main_Accords"]),
                            "Notes":            str(row["All_Notes"])[:200] + "…"
                                                if len(str(row["All_Notes"])) > 200
                                                else str(row["All_Notes"]),
                            "URL": row["URL"]
                        },
                        wishlist_key=f"disc_wish_{row['Name']}"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":

    st.title("Analytics")

    an_tab1, an_tab2 = st.tabs(["My Collection", "My Wishlist"])

    with an_tab1:

        liked        = collection[collection["Would I Wear?"].str.strip().str.lower() == "yes"]
        reviewed     = collection[collection["Review on Parfumo?"].astype(str).str.strip().str.lower().isin(["true", "yes", "1"])]
        full_bottles = collection[collection["Sample/Full Bottle?"].str.strip().str.lower() == "full bottle"]
        samples      = collection[collection["Sample/Full Bottle?"].str.strip().str.lower() == "sample"]
        miniatures   = collection[collection["Sample/Full Bottle?"].str.strip().str.lower() == "miniature"]

        # ── Row 1: Core metrics ───────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Fragrances",  len(collection))
        m2.metric("Would Wear",        f"{len(liked)} / {len(collection)}")
        m3.metric("Average Rating",    f"{collection['Rating'].mean():.2f}")
        m4.metric("5-Star Fragrances", int((collection["Rating"] == 5).sum()))

        # ── Row 2: Review + bottle tracking ──────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Reviews Written",   f"{len(reviewed)} / {len(collection)}")
        r2.metric("Full Bottles",      len(full_bottles))
        r3.metric("Samples",           len(samples))
        r4.metric("Miniatures",        len(miniatures))

        st.markdown("---")

        # ── Row 3: Rating + Season ────────────────────────────────────────────────
        left, right = st.columns(2)
        with left:
            st.subheader("Rating Distribution")
            st.bar_chart(collection["Rating"].value_counts().sort_index())
        with right:
            st.subheader("Season Breakdown")
            st.bar_chart(collection["Season"].value_counts())

        # ── Row 4: Strength + Gender ──────────────────────────────────────────────
        left2, right2 = st.columns(2)
        with left2:
            st.subheader("Strength Breakdown")
            st.bar_chart(collection["Strength"].value_counts())
        with right2:
            st.subheader("Gender Breakdown")
            st.bar_chart(collection["Gender"].value_counts())

        # ── Row 5: Bottle type + Would Wear ──────────────────────────────────────
        left3, right3 = st.columns(2)
        with left3:
            st.subheader("Bottle Type Breakdown")
            st.bar_chart(collection["Sample/Full Bottle?"].value_counts())
        with right3:
            st.subheader("Would Wear Breakdown")
            st.bar_chart(collection["Would I Wear?"].value_counts())

        st.markdown("---")

        # ── Brands ────────────────────────────────────────────────────────────────
        st.subheader("Brands in Your Collection")
        st.bar_chart(collection["Brand"].value_counts().head(20))

        st.markdown("---")

        # ── Notes + Themes ────────────────────────────────────────────────────────
        left4, right4 = st.columns(2)
        with left4:
            st.subheader("Top 20 Notes (liked fragrances)")
            note_counter = Counter()
            for _, row in liked.iterrows():
                for word in tokenize(row["Notes"]):
                    if len(word) > 2:
                        note_counter[word] += 1
            st.bar_chart(pd.DataFrame(
                note_counter.most_common(20), columns=["Note", "Count"]
            ).set_index("Note"))

        with right4:
            st.subheader("Top Themes (liked fragrances)")
            theme_counter = Counter()
            for _, row in liked.iterrows():
                for word in tokenize(row["Themes"]):
                    if len(word) > 2:
                        theme_counter[word] += 1
            st.bar_chart(pd.DataFrame(
                theme_counter.most_common(20), columns=["Theme", "Count"]
            ).set_index("Theme"))

        st.markdown("---")

        # ── Note co-occurrence heatmap ────────────────────────────────────────────
        st.subheader("Note Co-occurrence Heatmap")
        st.write("Shows which notes appear together most often in fragrances you'd wear.")

        heatmap_note_counter = Counter()
        for _, row in liked.iterrows():
            for word in tokenize(row["Notes"]):
                if len(word) > 2 and word not in NOTE_STOPWORDS:
                    heatmap_note_counter[word] += 1

        top_heatmap_notes = [w for w, _ in heatmap_note_counter.most_common(15)]

        if len(top_heatmap_notes) >= 2:
            n = len(top_heatmap_notes)
            matrix = [[0] * n for _ in range(n)]
            for _, row in liked.iterrows():
                row_notes = tokenize(row["Notes"])
                present = [i for i, note in enumerate(top_heatmap_notes) if note in row_notes]
                for i in present:
                    for j in present:
                        matrix[i][j] += 1
            import numpy as np
            mat = np.array(matrix, dtype=float)
            np.fill_diagonal(mat, 0)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(mat, cmap="YlOrRd")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(top_heatmap_notes, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(top_heatmap_notes, fontsize=9)
            for i in range(n):
                for j in range(n):
                    if mat[i][j] > 0:
                        ax.text(j, i, int(mat[i][j]), ha="center", va="center", fontsize=7, color="black")
            plt.colorbar(im, ax=ax, label="Times appearing together")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough notes data to build a heatmap.")

        st.markdown("---")
        st.subheader("Review Progress")
        review_pct = len(reviewed) / len(collection) * 100 if len(collection) > 0 else 0
        st.progress(int(review_pct))
        st.caption(f"{len(reviewed)} reviews written out of {len(collection)} fragrances ({review_pct:.0f}%)")

        not_reviewed = collection[
            ~collection["Review on Parfumo?"].astype(str).str.strip().str.lower().isin(["true", "yes", "1"])
        ][["Name", "Brand", "Rating"]].sort_values("Rating", ascending=False)

        with st.expander(f"Not yet reviewed ({len(not_reviewed)} fragrances)"):
            st.dataframe(not_reviewed)

    with an_tab2:

        if wishlist.empty:
            st.info("Your wishlist is empty — add some fragrances to see analytics.")
        else:
            wl_sub1, wl_sub2, wl_sub3 = st.tabs([
                "Wishlist Analytics", "Compatibility Ranking", "Similar to What You Own"
            ])

            with wl_sub1:
                st.write("How your wishlist compares to your collection.")

                wish_note_counter  = Counter()
                col_note_counter   = Counter()
                wish_theme_counter = Counter()
                col_theme_counter  = Counter()

                for _, wrow in wishlist.iterrows():
                    for word in tokenize(str(wrow.get("Notes", ""))):
                        if len(word) > 2 and word not in NOTE_STOPWORDS:
                            wish_note_counter[word] += 1
                    for word in tokenize(str(wrow.get("Themes", ""))):
                        if len(word) > 2 and word not in NOTE_STOPWORDS:
                            wish_theme_counter[word] += 1

                liked_an = collection[collection["Would I Wear?"].str.strip().str.lower() == "yes"]
                for _, crow in liked_an.iterrows():
                    for word in tokenize(crow["Notes"]):
                        if len(word) > 2 and word not in NOTE_STOPWORDS:
                            col_note_counter[word] += 1
                    for word in tokenize(crow["Themes"]):
                        if len(word) > 2 and word not in NOTE_STOPWORDS:
                            col_theme_counter[word] += 1

                wl_left, wl_right = st.columns(2)
                with wl_left:
                    st.subheader("Top Notes")
                    top_wish_notes = [w for w, _ in wish_note_counter.most_common(15)]
                    st.bar_chart(pd.DataFrame({
                        "Note":       top_wish_notes,
                        "Wishlist":   [wish_note_counter[w] for w in top_wish_notes],
                        "Collection": [col_note_counter.get(w, 0) for w in top_wish_notes],
                    }).set_index("Note"))

                with wl_right:
                    st.subheader("Top Themes")
                    top_wish_themes = [w for w, _ in wish_theme_counter.most_common(15)]
                    st.bar_chart(pd.DataFrame({
                        "Theme":      top_wish_themes,
                        "Wishlist":   [wish_theme_counter[w] for w in top_wish_themes],
                        "Collection": [col_theme_counter.get(w, 0) for w in top_wish_themes],
                    }).set_index("Theme"))

                new_notes  = set(wish_note_counter.keys()) - set(col_note_counter.keys())
                new_themes = set(wish_theme_counter.keys()) - set(col_theme_counter.keys())
                st.subheader("New Territory")
                st.write("Notes and themes on your wishlist that don't appear in your collection at all.")
                if new_notes:
                    st.write(f"**New notes:** {', '.join(sorted(new_notes))}")
                if new_themes:
                    st.write(f"**New themes:** {', '.join(sorted(new_themes))}")
                if not new_notes and not new_themes:
                    st.info("Your wishlist covers similar territory to your collection.")

            with wl_sub2:
                st.write("Wishlist items ranked by how well they match your taste profile.")
                note_profile_w, theme_profile_w = build_taste_profile(collection, idf)
                top_notes_w     = set(w for w, _ in note_profile_w.most_common(20))
                top_themes_w    = set(w for w, _ in theme_profile_w.most_common(10))
                profile_words_w = top_notes_w | top_themes_w

                wish_scores = []
                for _, wrow in wishlist.iterrows():
                    notes_words_w  = tokenize(str(wrow.get("Notes", "")))
                    themes_words_w = tokenize(str(wrow.get("Themes", "")))
                    score = 0
                    note_hits_w  = []
                    theme_hits_w = []
                    for word in profile_words_w:
                        word_idf = idf.get(word, 1.0)
                        if word in notes_words_w:
                            score += 4 * word_idf
                            note_hits_w.append(word)
                        if word in themes_words_w:
                            score += 2 * word_idf
                            theme_hits_w.append(word)
                    wish_scores.append({
                        "Name":          wrow["Name"],
                        "Brand":         wrow["Brand"],
                        "Score":         round(score, 1),
                        "Note Matches":  ", ".join(note_hits_w[:8]) or "—",
                        "Theme Matches": ", ".join(theme_hits_w[:5]) or "—",
                    })

                wish_ranked = pd.DataFrame(wish_scores).sort_values("Score", ascending=False)
                with st.container(height=500):
                    for _, r in wish_ranked.iterrows():
                        st.markdown(f"**{r['Name']}** — {r['Brand']}  |  Score: `{r['Score']}`")
                        if r["Note Matches"] != "—":
                            st.caption(f"Notes matching your taste: {r['Note Matches']}")
                        if r["Theme Matches"] != "—":
                            st.caption(f"Themes matching your taste: {r['Theme Matches']}")
                        st.markdown("---")

            with wl_sub3:
                st.write("Flags wishlist items that are very similar to fragrances you already own — so you don't buy something you essentially have.")
                SIMILARITY_THRESHOLD = 0.4

                sim_results = []
                for _, wrow in wishlist.iterrows():
                    wish_words = (
                        tokenize(str(wrow.get("Notes", ""))) |
                        tokenize(str(wrow.get("Themes", "")))
                    ) - NOTE_STOPWORDS

                    best_match = None
                    best_score = 0.0

                    for _, crow in collection.iterrows():
                        col_words = (
                            tokenize(crow["Notes"]) |
                            tokenize(crow["Themes"])
                        ) - NOTE_STOPWORDS

                        if not wish_words or not col_words:
                            continue

                        intersection = len(wish_words & col_words)
                        union        = len(wish_words | col_words)
                        sim          = intersection / union if union > 0 else 0

                        if sim > best_score:
                            best_score = sim
                            best_match = crow["Name"]

                    sim_results.append((wrow["Name"], best_score, best_match))

                sim_results.sort(key=lambda x: x[1], reverse=True)

                with st.container(height=500):
                    for wish_name, best_score, best_match in sim_results:
                        if best_score >= SIMILARITY_THRESHOLD:
                            st.warning(
                                f"**{wish_name}** is {int(best_score*100)}% similar to "
                                f"**{best_match}** which you already own."
                            )
                        else:
                            st.success(
                                f"**{wish_name}** — no very similar fragrance in your collection "
                                f"(closest: {best_match}, {int(best_score*100)}% similar)"
                            )

# ══════════════════════════════════════════════════════════════════════════════
# MY COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "My Collection":

    st.title("My Collection")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Browse Collection",
        "Edit / Delete",
        "Add Fragrance",
        "Wishlist"
    ])

    # =====================================================
    # BROWSE
    # =====================================================

    with tab1:

        search_collection = st.text_input(
            "Search Collection",
            placeholder="Name, brand, notes, themes..."
        )

        collection["search_text"] = (
            collection["Name"].astype(str)
            + " "
            + collection["Brand"].astype(str)
            + " "
            + collection["Notes"].astype(str)
            + " "
            + collection["Themes"].astype(str)
        ).str.lower()

        filtered = collection.copy()

        if search_collection:
            filtered = filtered[
                filtered["search_text"].str.contains(
                    search_collection.lower(),
                    na=False
                )
            ]

        display_cols = [c for c in filtered.columns if c != "search_text"]
        st.dataframe(filtered[display_cols])

        # ── Export ────────────────────────────────────────────────────────────
        # collection.to_csv(index=False) converts the dataframe to a CSV
        # string in memory — no file is written to disk. We pass that string
        # directly to the download button so clicking it saves to the user's
        # computer.
        st.download_button(
            label="⬇ Export collection as CSV",
            data=collection.to_csv(index=False),
            file_name="PerfumeCollection_export.csv",
            mime="text/csv"
        )

        with st.expander("⬆ Upload new collection CSV"):
            st.caption("The file must contain these columns: Name, Brand, Rating, Review on Parfumo?, Would I Wear?, Sample/Full Bottle?, Themes, Strength, Season, Gender, Notes.")
            uploaded_col = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                key=f"col_upload_{st.session_state.collection_upload_key}"
            )
            if uploaded_col is not None:
                try:
                    new_col_df = pd.read_csv(uploaded_col)
                    required_cols = {"Name", "Brand", "Rating", "Would I Wear?",
                                     "Sample/Full Bottle?", "Themes", "Strength",
                                     "Season", "Gender", "Notes"}
                    missing = required_cols - set(new_col_df.columns)
                    if missing:
                        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
                    else:
                        st.write(f"**{len(new_col_df)} fragrances** found. Preview:")
                        st.dataframe(new_col_df.head(5))
                        if st.button("Replace collection with this file", key="confirm_col_upload"):
                            backup_collection()
                            write_sheet("Collection", new_col_df)
                            if os.path.exists(COLLECTION_CACHE_PATH):
                                os.remove(COLLECTION_CACHE_PATH)
                            st.session_state.collection_version += 1
                            invalidate_data_cache()
                            st.session_state.collection_upload_key += 1
                            st.success("Collection replaced successfully.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    # =====================================================
    # EDIT / DELETE
    # =====================================================

    with tab2:

        fragrance_names = sorted(
            collection["Name"].tolist()
        )

        selected = st.selectbox(
            "Select fragrance or begin typing fragrance name",
            fragrance_names
        )

        row = collection[
            collection["Name"] == selected
        ].iloc[0]

        st.subheader("Edit")

        _edited_name = st.session_state.pop("edit_success", None)
        if _edited_name:
            st.success(f"'{_edited_name}' updated successfully.")

        ecol1, ecol2 = responsive_columns(2)

        with ecol1:
            name = st.text_input(
                "Name",
                value=row["Name"]
            )

            rating = st.select_slider(
                "Rating",
                options=[1,2,3,4,5],
                value=int(row["Rating"]),
                format_func=lambda x: "⭐" * x
            )

            would_wear = st.selectbox(
                "Would I Wear?",
                ["Yes", "No"],
                index=0 if str(row["Would I Wear?"]).lower() == "yes" else 1
            )

            season = st.selectbox(
                "Season",
                ["Any","Spring","Summer","Autumn","Winter"],
                index=[
                    "Any",
                    "Spring",
                    "Summer",
                    "Autumn",
                    "Winter"
                ].index(
                    str(row["Season"])
                )
            )

            themes = st.text_input(
                "Themes",
                value=row["Themes"]
            )

            reviewed = st.checkbox(
                "Reviewed on Parfumo?",
                value=bool(row["Review on Parfumo?"])
            )

        with ecol2:
            brand = st.text_input(
                "Brand",
                value=row["Brand"]
            )

            bottle = st.selectbox(
                "Sample/Full Bottle?",
                ["Sample", "Full Bottle", "Miniature"],
                index=[
                    "Sample",
                    "Full Bottle",
                    "Miniature"
                ].index(
                    str(row["Sample/Full Bottle?"])
                )
            )

            strength = st.selectbox(
                "Strength",
                ["Normal","Subtle","Strong"],
                index=[
                    "Normal",
                    "Subtle",
                    "Strong"
                ].index(
                    str(row["Strength"])
                )
            )

            gender = st.selectbox(
                "Gender",
                ["Unisex","Masculine","Feminine"],
                index=[
                    "Unisex",
                    "Masculine",
                    "Feminine"
                ].index(
                    str(row["Gender"])
                )
            )

            notes = st.text_area(
                "Notes",
                value=row["Notes"],
                height=120
            )

        col1, col2 = responsive_columns(2)

        with col1:

            if st.button("Save Changes"):

                match = collection[collection["Name"] == selected]
                if len(match) == 0:
                    st.error("Fragrance not found.")
                else:
                    # ── Validation ─────────────────────────────────────────────
                    save_errors = []
                    if not str(name).strip():
                        save_errors.append("Name cannot be blank.")
                    if not str(brand).strip():
                        save_errors.append("Brand cannot be blank.")
                    # Duplicate check — allow keeping the same name (editing in place)
                    existing_names = collection["Name"].str.strip().str.lower().tolist()
                    if (str(name).strip().lower() != selected.lower()
                            and str(name).strip().lower() in existing_names):
                        save_errors.append(f"'{name}' already exists in your collection.")

                    if save_errors:
                        for e in save_errors:
                            st.error(e)
                    else:
                        idx = match.index[0]
                        collection.loc[idx, "Name"]                = name
                        collection.loc[idx, "Brand"]               = brand
                        collection.loc[idx, "Rating"]              = rating
                        collection.loc[idx, "Review on Parfumo?"]  = reviewed
                        collection.loc[idx, "Would I Wear?"]       = would_wear
                        collection.loc[idx, "Sample/Full Bottle?"] = bottle
                        collection.loc[idx, "Themes"]              = themes
                        collection.loc[idx, "Strength"]            = strength
                        collection.loc[idx, "Season"]              = season
                        collection.loc[idx, "Gender"]              = gender
                        collection.loc[idx, "Notes"]               = notes

                        backup_collection()
                        write_sheet("Collection", collection)
                        st.session_state.collection_version += 1
                        invalidate_data_cache()

                        if os.path.exists(COLLECTION_CACHE_PATH):
                            os.remove(COLLECTION_CACHE_PATH)

                        st.session_state.edit_success = name
                        st.rerun()

        with col2:

            if st.session_state.get("confirm_delete") == selected:
                st.warning(f"Are you sure you want to delete **{selected}**? This cannot be undone.")
                if st.button("Yes, delete"):
                    backup_collection()
                    updated = collection[collection["Name"] != selected]
                    write_sheet("Collection", updated)
                    st.session_state.collection_version += 1
                    invalidate_data_cache()
                    st.session_state.confirm_delete = None
                    if os.path.exists(COLLECTION_CACHE_PATH):
                        os.remove(COLLECTION_CACHE_PATH)
                    st.success(f"{selected} deleted.")
                    st.rerun()
                if st.button("Cancel"):
                    st.session_state.confirm_delete = None
                    st.rerun()
            else:
                if st.button("Delete Fragrance"):
                    st.session_state.confirm_delete = selected
                    st.rerun()

    # =====================================================
    # ADD FRAGRANCE
    # =====================================================

    with tab3:

        st.subheader("Add Fragrance")

        if st.session_state.get("last_add_success"):
            st.success(st.session_state.pop("last_add_success"))

        if st.session_state.clear_add_form_pending:
            st.session_state.clear_add_form_pending = False
            for _k in ("_add_form_name", "_add_form_brand", "_add_form_notes", "_add_form_themes"):
                st.session_state[_k] = ""

        if "wishlist_to_add" in st.session_state:
            _pname = st.session_state.pop("wishlist_to_add")
            st.session_state["_add_form_name"]   = _pname
            st.session_state["_add_form_brand"]  = st.session_state.pop("wishlist_to_add_brand", "")
            st.session_state["_add_form_notes"]  = st.session_state.pop("wishlist_to_add_notes", "")
            st.session_state["_add_form_themes"] = st.session_state.pop("wishlist_to_add_themes", "")
            st.session_state["_add_form_notice"] = _pname

        _notice = st.session_state.pop("_add_form_notice", "")
        if _notice:
            st.info(f"Pre-filled from your wishlist: **{_notice}**. Fill in the remaining fields and click **Add Fragrance**.")

        with st.expander("Add from Parfumo URL"):
            add_parfumo_url = st.text_input(
                "Paste a Parfumo page URL",
                placeholder="https://www.parfumo.com/Perfumes/...",
                key=f"add_parfumo_url_input_{st.session_state.add_parfumo_url_key}"
            )
            if st.button("Fetch from URL", key="add_parfumo_fetch_btn"):
                if not add_parfumo_url.strip():
                    st.error("Please paste a URL first.")
                else:
                    with st.spinner("Fetching page…"):
                        _data, _err = scrape_parfumo_url(add_parfumo_url.strip())
                    if _err:
                        st.error(f"Could not scrape page: {_err}")
                    else:
                        st.session_state["_add_form_name"]   = _data.get("Name", "")
                        st.session_state["_add_form_brand"]  = _data.get("Brand", "")
                        st.session_state["_add_form_themes"] = _data.get("Themes", "")
                        st.session_state["_add_form_notes"]  = _data.get("Notes", "")
                        st.session_state.add_parfumo_url_key += 1
                        st.rerun()

        with st.form(f"add_fragrance_form_{st.session_state.add_form_key}"):

            acol1, acol2 = responsive_columns(2)

            with acol1:
                name = st.text_input("Name", key="_add_form_name")

                rating = st.select_slider(
                    "Rating",
                    options=[1,2,3,4,5],
                    value=3,
                    format_func=lambda x: "⭐" * x
                )

                would_wear = st.selectbox(
                    "Would I Wear?",
                    ["Yes", "No"]
                )

                season = st.selectbox(
                    "Season",
                    ["Any","Spring","Summer","Autumn","Winter"]
                )

                themes = st.text_input("Themes", key="_add_form_themes")

                reviewed = st.checkbox(
                    "Reviewed on Parfumo?"
                )

            with acol2:
                brand = st.text_input("Brand", key="_add_form_brand")

                bottle = st.selectbox(
                    "Sample/Full Bottle?",
                    ["Sample","Full Bottle","Miniature"]
                )

                strength = st.selectbox(
                    "Strength",
                    ["Normal","Subtle","Strong"]
                )

                gender = st.selectbox(
                    "Gender",
                    ["Unisex","Masculine","Feminine"]
                )

                notes = st.text_area("Notes", key="_add_form_notes", height=120)


            btn_col1, btn_col2 = responsive_columns(2)
            with btn_col1:
                submitted = st.form_submit_button("Add Fragrance")
            with btn_col2:
                clear_clicked = st.form_submit_button("Clear Form")

            if clear_clicked:
                st.session_state.clear_add_form_pending = True
                st.session_state.add_form_key += 1
                st.rerun()

            if submitted:
                # ── Validation ─────────────────────────────────────────────────
                errors = []
                if not str(name).strip():
                    errors.append("Name is required.")
                if not str(brand).strip():
                    errors.append("Brand is required.")
                if not str(themes).strip():
                    errors.append("Themes is required.")
                if not str(notes).strip():
                    errors.append("Notes is required.")
                if str(name).strip() and str(name).strip().lower() in collection["Name"].str.strip().str.lower().tolist():
                    errors.append(f"'{name}' already exists in your collection.")

                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    new_row = pd.DataFrame([{
                        "Name":                name,
                        "Brand":               brand,
                        "Rating":              rating,
                        "Review on Parfumo?":  reviewed,
                        "Would I Wear?":       would_wear,
                        "Sample/Full Bottle?": bottle,
                        "Themes":              themes,
                        "Strength":            strength,
                        "Season":              season,
                        "Gender":              gender,
                        "Notes":               notes
                    }])

                    backup_collection()
                    updated = pd.concat([collection, new_row], ignore_index=True)
                    write_sheet("Collection", updated)

                    st.session_state.add_form_key += 1
                    st.session_state.collection_version += 1
                    st.session_state.last_add_success = f"'{name}' added to your collection!"

                    st.session_state.clear_add_form_pending = True

                    if os.path.exists(COLLECTION_CACHE_PATH):
                        os.remove(COLLECTION_CACHE_PATH)

                    wishlist_names = wishlist["Name"].str.strip().str.lower().tolist()
                    if str(name).strip().lower() in wishlist_names:
                        st.session_state.wishlist_remove_prompt = name.strip()

                    invalidate_data_cache()
                    st.rerun()

        if st.session_state.wishlist_remove_prompt:
            prompt_name = st.session_state.wishlist_remove_prompt
            st.info(f"**'{prompt_name}'** is on your wishlist. Would you like to remove it now?")
            pcol1, pcol2 = responsive_columns(2)
            with pcol1:
                if st.button("Yes, remove from wishlist", key="wishlist_remove_yes"):
                    updated_wish = wishlist[
                        wishlist["Name"].str.strip().str.lower() != prompt_name.lower()
                    ]
                    write_sheet("Wishlist", updated_wish)
                    st.session_state.wishlist_version += 1
                    invalidate_data_cache()
                    st.session_state.wishlist_remove_prompt = None
                    st.success(f"'{prompt_name}' removed from your wishlist.")
                    st.rerun()
            with pcol2:
                if st.button("No, keep it", key="wishlist_remove_no"):
                    st.session_state.wishlist_remove_prompt = None
                    st.rerun()

    # =====================================================
    # WISHLIST TAB
    # =====================================================

    with tab4:

        st.subheader("My Wishlist")
        st.write("Fragrances you want to try. Search the Parfumo database to add them, or add manually.")

        # ── Search Parfumo to add to wishlist ─────────────────────────────────
        wish_search = st.text_input("Search Parfumo database by name or brand", placeholder="e.g. Santal 33 or Chanel", key="wish_search_input")

        if wish_search:
            _words = [w for w in wish_search.lower().split() if w]
            _mask = pd.Series([True] * len(parfumo), index=parfumo.index)
            for _w in _words:
                _mask &= (
                    parfumo["Name"].str.contains(_w, case=False, na=False)
                    | parfumo["Brand"].str.contains(_w, case=False, na=False)
                )
            parfumo_matches = parfumo[_mask]

            if parfumo_matches.empty:
                st.info("No matches found in the Parfumo database.")
            else:
                total = len(parfumo_matches)
                st.caption(f"{total} result{'s' if total != 1 else ''} found")
                with st.container(height=420):
                    for idx, prow in parfumo_matches.iterrows():
                        p_rating = prow["Rating_Value"]
                        rating_display = f"{float(p_rating):.1f}/10" if pd.notna(p_rating) else "Not rated"
                        st.write(f"**{prow['Name']}** — {prow['Brand']}  |  {rating_display}")
                        if str(prow["Main_Accords"]).strip() and prow["Main_Accords"] != "nan":
                            st.caption(f"Themes: {str(prow['Main_Accords'])[:150]}")
                        st.caption(f"Notes: {str(prow['All_Notes'])[:150]}")
                        if str(prow.get("URL", "")).strip():
                            st.caption(prow["URL"])

                        if prow["Name"].strip().lower() in wishlist["Name"].str.strip().str.lower().tolist():
                            st.caption("✓ Already on your wishlist")
                        elif st.button(f"+ Add to Wishlist", key=f"wish_add_{idx}_{prow['Name']}"):
                            new_wish = pd.DataFrame([{
                                "Name":   prow["Name"],
                                "Brand":  prow["Brand"],
                                "Notes":  prow["All_Notes"],
                                "Themes": prow["Main_Accords"],
                                "URL":    prow["URL"],
                            }])
                            updated_wish = pd.concat([wishlist, new_wish], ignore_index=True)
                            write_sheet("Wishlist", updated_wish)
                            st.session_state.wishlist_version += 1
                            invalidate_data_cache()
                            st.toast(f"'{prow['Name']}' added to your wishlist!")
                            st.rerun()

        st.markdown("---")

        # ── Add from Parfumo URL ───────────────────────────────────────────────
        with st.expander("Add from Parfumo URL"):
            parfumo_url = st.text_input(
                "Paste a Parfumo page URL",
                placeholder="https://www.parfumo.com/Perfumes/...",
                key=f"parfumo_url_input_{st.session_state.parfumo_url_key}"
            )
            if st.button("Fetch from URL", key="parfumo_fetch_btn"):
                if not parfumo_url.strip():
                    st.error("Please paste a URL first.")
                else:
                    with st.spinner("Fetching page…"):
                        data, err = scrape_parfumo_url(parfumo_url.strip())
                    if err:
                        st.error(f"Could not scrape page: {err}")
                    else:
                        st.session_state.scraped_parfumo = data
                        st.session_state.scraped_parfumo_form_key += 1

            if st.session_state.scraped_parfumo:
                s = st.session_state.scraped_parfumo
                st.info("Review and edit the details below before adding.")
                with st.form(f"scraped_parfumo_form_{st.session_state.scraped_parfumo_form_key}"):
                    sc_name   = st.text_input("Name",   value=s.get("Name", ""))
                    sc_brand  = st.text_input("Brand",  value=s.get("Brand", ""))
                    sc_themes = st.text_input("Themes", value=s.get("Themes", ""))
                    sc_notes  = st.text_area("Notes",   value=s.get("Notes", ""), height=80)
                    sc_url    = st.text_input("URL",    value=s.get("URL", ""))
                    btn_col1, btn_col2 = responsive_columns([1, 1])
                    with btn_col1:
                        sc_submit = st.form_submit_button("Add to Wishlist")
                    with btn_col2:
                        sc_clear = st.form_submit_button("Clear Form")
                    if sc_clear:
                        st.session_state.scraped_parfumo = None
                        st.session_state.scraped_parfumo_form_key += 1
                        st.session_state.parfumo_url_key += 1
                        st.rerun()
                    if sc_submit:
                        if not str(sc_name).strip():
                            st.error("Name is required.")
                        elif sc_name.strip().lower() in wishlist["Name"].str.strip().str.lower().tolist():
                            st.error(f"'{sc_name}' is already on your wishlist.")
                        else:
                            new_wish = pd.DataFrame([{
                                "Name":   sc_name.strip(),
                                "Brand":  sc_brand.strip(),
                                "Themes": sc_themes.strip(),
                                "Notes":  sc_notes.strip(),
                                "URL":    sc_url.strip(),
                            }])
                            updated_wish = pd.concat([wishlist, new_wish], ignore_index=True)
                            write_sheet("Wishlist", updated_wish)
                            st.session_state.wishlist_version += 1
                            invalidate_data_cache()
                            st.session_state.scraped_parfumo = None
                            st.session_state.scraped_parfumo_form_key += 1
                            st.session_state.parfumo_url_key += 1
                            st.toast(f"'{sc_name}' added to your wishlist!")
                            st.rerun()

        # ── Manual add ────────────────────────────────────────────────────────
        with st.expander("Add manually (not in Parfumo database)"):
            with st.form(f"manual_wish_form_{st.session_state.manual_wish_form_key}"):
                mw_name   = st.text_input("Name")
                mw_brand  = st.text_input("Brand")
                mw_themes = st.text_input("Themes (optional)")
                mw_notes  = st.text_input("Notes (optional)")
                mw_url    = st.text_input("URL (optional)")
                mw_submit = st.form_submit_button("Add to Wishlist")
                if mw_submit:
                    if not str(mw_name).strip():
                        st.error("Name is required.")
                    elif mw_name.strip().lower() in wishlist["Name"].str.strip().str.lower().tolist():
                        st.error(f"'{mw_name}' is already on your wishlist.")
                    else:
                        new_wish = pd.DataFrame([{
                            "Name":   mw_name.strip(),
                            "Brand":  mw_brand.strip(),
                            "Themes": mw_themes.strip(),
                            "Notes":  mw_notes.strip(),
                            "URL":    mw_url.strip(),
                        }])
                        updated_wish = pd.concat([wishlist, new_wish], ignore_index=True)
                        write_sheet("Wishlist", updated_wish)
                        st.session_state.wishlist_version += 1
                        invalidate_data_cache()
                        st.session_state.manual_wish_form_key += 1
                        st.toast(f"'{mw_name}' added to your wishlist!")
                        st.rerun()

        with st.expander("⬆ Upload new wishlist CSV"):
            st.caption("The file must contain these columns: Name, Brand, Notes, Themes, URL.")
            uploaded_wish = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                key=f"wish_upload_{st.session_state.wishlist_upload_key}"
            )
            if uploaded_wish is not None:
                try:
                    new_wish_df = pd.read_csv(uploaded_wish)
                    new_wish_df = new_wish_df.drop(columns=["Rating_Value"], errors="ignore")
                    if "Main_Accords" in new_wish_df.columns and "Themes" not in new_wish_df.columns:
                        new_wish_df = new_wish_df.rename(columns={"Main_Accords": "Themes"})
                    required_wish_cols = {"Name", "Brand", "Notes", "Themes", "URL"}
                    missing = required_wish_cols - set(new_wish_df.columns)
                    if missing:
                        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
                    else:
                        st.write(f"**{len(new_wish_df)} fragrances** found. Preview:")
                        st.dataframe(new_wish_df.head(5))
                        if st.button("Replace wishlist with this file", key="confirm_wish_upload"):
                            write_sheet("Wishlist", new_wish_df)
                            st.session_state.wishlist_version += 1
                            invalidate_data_cache()
                            st.session_state.wishlist_upload_key += 1
                            st.success("Wishlist replaced successfully.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        # ── Current wishlist ──────────────────────────────────────────────────
        st.subheader(f"Your Wishlist ({len(wishlist)} fragrances)")

        if not wishlist.empty:
            _wish_b64 = base64.b64encode(wishlist.to_csv(index=False).encode()).decode()
            st.markdown(
                f'<style>'
                f'.wish-dl-btn{{display:inline-flex;align-items:center;justify-content:center;'
                f'padding:0.25rem 0.75rem;background-color:transparent;border-radius:0.5rem;'
                f'text-decoration:none!important;cursor:pointer;font-size:1rem;line-height:1.6;font-weight:400;'
                f'border:1px solid rgba(49,51,63,0.2);color:rgb(49,51,63)!important;}}'
                f'@media(prefers-color-scheme:dark){{'
                f'.wish-dl-btn{{border:1px solid rgba(250,250,250,0.2);color:rgb(250,250,250)!important;}}}}'
                f'</style>'
                f'<a href="data:file/csv;base64,{_wish_b64}" download="Wishlist_export.csv" class="wish-dl-btn">'
                f'⬇ Export wishlist as CSV</a>',
                unsafe_allow_html=True
            )

        if wishlist.empty:
            st.info("Your wishlist is empty. Search above to add fragrances.")
        else:
            wish_filter = st.text_input("Search wishlist", placeholder="Name, brand, themes, notes…", key="wish_list_filter")
            filtered_wishlist = wishlist[
                (wishlist["Name"].str.contains(wish_filter, case=False, na=False))
                | (wishlist["Brand"].str.contains(wish_filter, case=False, na=False))
                | (wishlist["Themes"].str.contains(wish_filter, case=False, na=False))
                | (wishlist["Notes"].str.contains(wish_filter, case=False, na=False))
            ] if wish_filter else wishlist
            with st.container(height=500):
                for _, wrow in filtered_wishlist.iterrows():
                    wcol1, wcol2 = responsive_columns([4, 1])
                    with wcol1:
                        st.write(f"**{wrow['Name']}** — {wrow['Brand']}")
                        if str(wrow.get("Themes", "")).strip():
                            st.caption(f"Themes: {str(wrow['Themes'])[:150]}")
                        if str(wrow.get("Notes", "")).strip():
                            st.caption(f"Notes: {str(wrow['Notes'])[:150]}")
                        if str(wrow.get("URL", "")).strip():
                            st.caption(wrow["URL"])
                        if st.session_state.get("wishlist_add_notice") == wrow["Name"]:
                            st.info(f"Go to the **Add Fragrance** tab to add **{wrow['Name']}** to your collection.")
                            del st.session_state["wishlist_add_notice"]

                    with wcol2:
                        if st.button("I got it!", key=f"got_{wrow['Name']}"):
                            st.session_state.wishlist_to_add = wrow["Name"]
                            st.session_state.wishlist_to_add_brand = str(wrow.get("Brand", ""))
                            st.session_state.wishlist_to_add_notes = str(wrow.get("Notes", ""))
                            st.session_state.wishlist_to_add_themes = str(wrow.get("Themes", ""))
                            st.session_state.add_form_key += 1
                            st.session_state.wishlist_add_notice = wrow["Name"]
                            st.rerun()

                        if st.button("Remove", key=f"wish_remove_{wrow['Name']}"):
                            updated_wish = wishlist[wishlist["Name"] != wrow["Name"]]
                            write_sheet("Wishlist", updated_wish)
                            st.session_state.wishlist_version += 1
                            invalidate_data_cache()
                            st.rerun()

                    st.markdown("---")
