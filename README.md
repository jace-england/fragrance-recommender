# Fragrance Collection and Recommendation Tool

I'm really into niche fragrances, and I use Parfumo to log and review my collection. I've been using my Notes app to keep track of which fragrances are on my wishlist, which ones I have samples for, and which ones I still need to write Parfumo reviews for. But the page is getting really long and it's a pain to scroll through everything... the Parfumo website also doesn't have every fragrance I own in their database, so I wanted to create an app that collects everything into one place.

I created a personal fragrance collection and recommendation tool built with Streamlit, designed to help manage a perfume collection, discover new fragrances, create a wishlist and get daily wear recommendations based on mood, season, and taste profile.

It combines rule-based scoring, TF-IDF weighted matching, and semantic search (Sentence Transformers) to surface personalised recommendations from both a personal collection and the Parfumo fragrance database. It also has the option to add wishlist entries manually, through the Parfumo dataset or via Parfumo URL (with scraper that handles pyramid and single-section note formats, filters usernames and junk).

## Goals

- Make it easy to manage, track, and grow a fragrance collection and wishlist in one place
- Build a personalised recommendation engine that learns taste from an existing collection
- Provide meaningful discovery of new fragrances based on what you already love
- Surface analytical insights about collection composition and wishlist metrics

## Key Features

### Wear Today
- Semantic search — describe a mood or vibe in plain language
- Keyword matching on specific notes and themes
- Season, strength, and masculine/feminine/unisex filters
- Blended scoring combining semantic similarity and keyword relevance
- **Surprise Me** — random pick from your top-rated wearable fragrances
- Explainable recommendations with "Why it matched" summaries

### Discover New Fragrances
- Taste profile learned from your collection using IDF-weighted note scoring
- Brand affinity boosting for fragrance houses you already love
- Keyword and semantic refine search
- **Similar to This** — find Parfumo fragrances semantically similar to ones you already own
- Filters out fragrances you own or already have on your wishlist
- Minimum community rating threshold to remove obscure results

### My Collection
- Browse, search, edit, and delete entries
    - Add manually, from wishlist, or via Parfumo URL with automatic note/theme/accent population via scraping
    - URL scraper handles both pyramid (top/heart/base) and single-section note formats
    - Filters usernames and page junk from scraped notes
- Data validation on save (duplicate detection, required field checks)
- Auto-backup to `PerfumeCollection_backup.csv` before every write
- CSV export and import

### Wishlist
- Add from:
    - Parfumo database search
    - Inputting a Parfumo URL with automatic note/theme/accent population
    - Manual input form
- "I got it!" flow to move a wishlist item into your collection through automatic population in the Add Fragrance tab
- Wishlist table is stored separately in `Wishlist.csv`
- CSV export and import

### Analytics
- Collection metrics: total, would wear vs. not, average rating, 5-star count, review progress
- Breakdown charts: rating, season, strength, gender, bottle type
- Top notes and themes (for liked fragrances only)
- Note co-occurrence heatmap
- Review progress tracker with expandable list of unreviewed fragrances
- Wishlist analytics: note/theme comparison vs collection, compatibility ranking, similarity to owned fragrances

## Recommendation Engine

The scoring system combines three signals:

**Keyword scoring (Wear Today & Discover)**
- Notes matches weighted at 4×, themes matches weighted at 2×
- IDF weighting reduces the influence of generic notes (vanilla, musk, amber) that appear in thousands of fragrances, boosting distinctive notes unique to the user's taste

**Brand affinity (Discover)**
- Brands from liked collection fragrances receive a score multiplier up to 2× based on how many fragrances the user owns and their ratings

**Semantic search (Wear Today & Discover)**
- `all-MiniLM-L6-v2` via Sentence Transformers encodes fragrance descriptions and queries into 384-dimensional embeddings
- Cosine similarity scores are blended with keyword scores (60/40 by default)
- Parfumo embeddings are pre-computed and cached to disk at first run

## Data

- `PerfumeCollection.csv` — personal fragrance collection
- `parfumo_data_clean.csv` — scraped Parfumo database (~59,000 fragrances, 1,451 brands, 1709 to 2024) sourced from [here](https://www.kaggle.com/datasets/ibrahimqasimi/parfumo-perfume-database-59k-fragrances)
- `Wishlist.csv` — generated upon first wishlist entry
- `parfumo_embeddings.npy` — pre-computed embeddings (generated locally, not committed to repo)
- `collection_embeddings.npy` — pre-computed collection embeddings (generated locally, not committed to repo)

## Tools Used

- Python (pandas, numpy, scikit-learn, matplotlib)
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- BeautifulSoup (Parfumo URL scraper)
- Visual Studio Code

## Project Structure

```
fragrance-recommender/

├── app_fixed.py              # Main Streamlit application
├── PerfumeCollection.csv     # Personal fragrance collection
├── parfumo_data_clean.csv    # Parfumo scraped database
├── Wishlist.csv              # Wishlist (auto-generated)
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Pre-compute Parfumo embeddings (one-time, takes a few minutes)
.venv/bin/python -c "
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('parfumo_data_clean.csv')
def combine_notes(row):
    parts = [str(row['Top_Notes']), str(row['Middle_Notes']), str(row['Base_Notes'])]
    return ', '.join(p for p in parts if p.lower() != 'nan')
df['All_Notes'] = df.apply(combine_notes, axis=1)
docs = [(str(row['Main_Accords']) + '. ' + row['All_Notes']).strip('. ') for _, row in df.iterrows()]
embs = model.encode(docs, batch_size=256, show_progress_bar=True)
np.save('parfumo_embeddings.npy', embs)
print('Done!')
"

# Run the app
.venv/bin/python -m streamlit run app_fixed.py
```

## Project Outcome

This project demonstrates:

- End-to-end development of a multi-page Streamlit application from scratch
- Implementation of a recommendation engine combining rule-based scoring, TF-IDF weighting, and transformer-based semantic search. 
- Using my Discover New Fragrances tool, I was actually recommended fragrances that I already had in my Notes app wishlist which was a great proof of concept!
- Use of NLP techniques (tokenisation, IDF, cosine similarity, sentence embeddings)
- Web scraping with BeautifulSoup including robust parsing logic for inconsistent page structures
- Data persistence, caching strategy, and session state management in a Streamlit app
- Mobile-responsive UI design within Streamlit's constraints
- Iterative, sprint-based development — building core functionality first and layering complexity progressively

## Ways to Improve

- **Wear history tracking** — log which fragrance is worn each day and apply recency weighting so frequently worn fragrances are gently down-scored to encourage rediscovery
- **Persistent cloud storage** — replace CSV file storage with a lightweight database (e.g. Supabase or Google Sheets API) so collection edits made through the app persist between Streamlit Cloud redeploys
- **Upgrade to modern Streamlit** — migrate `@st.cache` to `@st.cache_data` / `@st.cache_resource` and adopt `st.data_editor` for inline collection editing
- **Notes autocomplete** — suggest known note names as the user types when adding a fragrance manually, reducing inconsistent note naming across entries
- **Fragrance layering suggestions** — identify pairs in the collection that share complementary notes and suggest wearing them together
- **Cost tracking** — add purchase price and bottle size fields to calculate cost per ml and total collection value
- **Richer dataset** — integrate a more current fragrance database or supplement the existing Parfumo dataset with newer releases