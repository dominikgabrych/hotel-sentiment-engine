"""
pb4_visualize.py — Aspect Category Detection & Visualization (PB4)

Phase 2: Final categorization, filtering, and generation of Polish-labeled 
charts for the thesis. Evaluates the 'OTHER' category with an expanded dictionary.

Generated outputs:
- fig_05_absa_negative_aspects.png (Bar chart of complaint frequencies)
- fig_06_absa_wordcloud_per_aspect.png (Context n-grams grid)
- fig_07_absa_sentiment_heatmap.png (Hotel Stars vs. Complaint Category)
- tab_06_absa_aspect_counts.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud

# --- AGENT-EXPANDED DICTIONARY ---
BASE_DICTIONARY = {
    "STAFF": [
        "staff", "reception", "service", "employee", "helpful", "rude", "friendly", 
        "lady", "guy", "manager", "desk", "personnel", "waiter", "maid",
        "gentleman", "ladies", "served"
    ],
    "CLEANLINESS": [
        "clean", "dirty", "smell", "hygiene", "bathroom", "towel", "sheet", "dust", 
        "floor", "stain", "mold", "spider", "housekeep", "trash", "bin", "hair", "mess",
        "toilet", "sink", "bathtub", "soap", "carpet"
    ],
    "COMFORT": [
        "bed", "noise", "temperature", "pillow", "mattress", "quiet", "sleep", "cold", 
        "hot", "air", "ac ", "blanket", "loud", "warm", "heat", "freeze", "comfort", 
        "wall", "sound", "ventilation", "music", "fan"
    ],
    "FACILITIES": [
        "gym", "pool", "wifi", "elevator", "parking", "amenit", "breakfast", "restaurant", 
        "food", "shower", "bar", "tv", "kettle", "internet", "lift", "spa", "sauna", 
        "water", "coffee", "tea", "drain", "socket", "light", "door", "window",
        "room", "furniture", "decor", "curtain", "furnishing", "tile", "ceiling", 
        "equipment", "refrigerator", "fridge", "bread", "egg", "dish", "selection", 
        "table", "sofa", "corridor", "facility"
    ],
    "VALUE": [
        "price", "expensive", "worth", "overprice", "money", "cost", "budget", "cheap", 
        "value", "pay", "quality"
    ],
    "LOCATION": [
        "location", "near", "distance", "center", "transport", "access", "view", 
        "tram", "bus", "station", "train", "airport", "area", "street"
    ]
}

# --- CHART LABELS (POLISH FRONT-END) ---
POLISH_LABELS = {
    "STAFF": "Personel i Obsługa",
    "CLEANLINESS": "Czystość i Higiena",
    "COMFORT": "Komfort i Akustyka",
    "FACILITIES": "Udogodnienia i Pokój",
    "VALUE": "Stosunek Jakości do Ceny",
    "LOCATION": "Lokalizacja",
    "OTHER": "Inne (pominięte)"
}

def map_aspect_to_category(aspect_term: str) -> str:
    """Maps an aspect phrase to the main category using the expanded dictionary."""
    if not isinstance(aspect_term, str):
        return "OTHER"
    
    aspect_lower = aspect_term.lower()
    
    for category, keywords in BASE_DICTIONARY.items():
        for keyword in keywords:
            if keyword in aspect_lower:
                return category
    return "OTHER"

def generate_outputs() -> None:
    print("=" * 60)
    print("STARTING PB4 VISUALIZATION GENERATION (ABSA)")
    print("=" * 60)
    
    # Create required directories
    os.makedirs("data/04_results/tables", exist_ok=True)
    os.makedirs("data/04_results/figures", exist_ok=True)
    
    # 1. Load and map data
    df = pd.read_csv("data/04_results/absa_results.csv")
    df["aspect_category"] = df["extracted_aspect"].apply(map_aspect_to_category)
    
    # Translate labels for Polish charting
    df["kategoria_pl"] = df["aspect_category"].map(POLISH_LABELS)
    
    # 2. Generate and save the main statistical table (PB4 tab_06)
    tab_06 = df.groupby(["kategoria_pl", "sentiment"]).size().unstack(fill_value=0).reset_index()
    tab_06.columns.name = None
    tab_06 = tab_06.sort_values(by="NEGATIVE", ascending=False)
    tab_06.to_csv("data/04_results/tables/tab_06_absa_aspect_counts.csv", index=False, encoding="utf-8")
    print("[✓] tab_06_absa_aspect_counts.csv saved.")

    # Filter strictly for NEGATIVE sentiment, ignoring 'OTHER' category mapping failures
    df_neg = df[(df["sentiment"] == "NEGATIVE") & (df["aspect_category"] != "OTHER")]
    
    # Global chart styling
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": ".8", "font.size": 11, "font.family": "sans-serif"})

    # -------------------------------------------------------------------------
    # PLOT 05: Ranking of Negative Aspects (Bar Chart)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    counts = df_neg["kategoria_pl"].value_counts()
    order = counts.index
    
    ax = sns.countplot(y="kategoria_pl", data=df_neg, order=order, hue="kategoria_pl", palette="rocket", legend=False)
    
    plt.title("Najczęstsze kategorie skarg gości (Analiza Sentymentu - ABSA)", fontsize=14, pad=15)
    plt.xlabel("Liczba negatywnych wzmianek w opiniach", fontsize=12)
    plt.ylabel("Kategoria", fontsize=12)
    
    # Set limit explicitly to retain border but prevent drawing an extra axis label
    ax.set_xlim(0, counts.max() + 250)
    
    for container in ax.containers:
        ax.bar_label(container, padding=5)
        
    plt.tight_layout()
    plt.savefig("data/04_results/figures/fig_05_absa_negative_aspects.png", dpi=300)
    plt.close()
    print("[✓] fig_05_absa_negative_aspects.png saved.")

    # -------------------------------------------------------------------------
    # PLOT 06: Bigram Wordcloud (Pseudo-OTE representation)
    # -------------------------------------------------------------------------
    # Read original text to extract adjectives surrounding the extracted aspects
    df_input = pd.read_csv("data/03_processed/absa_input.csv")
    
    # Deduplicate by source_id to prevent cartesian duplication during the merge
    df_input_unique = df_input.drop_duplicates(subset=["source_id"])
    df_neg_full = pd.merge(df_neg, df_input_unique[["source_id", "neg_text"]], on="source_id", how="left")
    
    categories = [cat for cat in BASE_DICTIONARY.keys()]
    cat_titles = {c: POLISH_LABELS[c] for c in categories}
    
    # Basic stopwords to avoid meaningless grammatical structures in collocations
    boring_words = {"the", "a", "an", "is", "was", "were", "are", "of", "to", "in", "on", "for", 
                    "with", "and", "but", "very", "too", "so", "it", "this", "that", "my", "our", 
                    "room", "hotel", "at", "from", "as", "be", "we", "they", "there", "i", "you",
                    "not", "no", "came", "could", "couldn", "did", "didn", "does", "had", "has", 
                    "here", "if", "maybe", "me", "one", "only", "or", "out", "over", "perhaps", 
                    "s", "said", "same", "she", "should", "soem", "took", "wasn", "which", "who", 
                    "would", "etc", "also", "try", "tried", "asked", "despite", "increases", 
                    "puro", "chair", "everything", "says", "unfamiliar", "shortage", "without", 
                    "shared", "young"}
                    
    PL_TRANSLATION_DICT = {
        'air': 'powietrze', 'area': 'okolica', 'bad': 'zły', 'bathroom': 'łazienka', 'bathrooms': 'łazienki',
        'bathtub': 'wanna', 'bed': 'łóżko', 'bedding': 'pościel', 'beds': 'łóżka', 'bread': 'chleb',
        'breakfast': 'śniadanie', 'breakfasts': 'śniadania', 'broken': 'zepsuty', 'carpet': 'dywan',
        'carpets': 'dywany', 'center': 'centrum', 'cheat': 'oszustwo', 'clean': 'czystość', 'cleaning': 'sprzątanie',
        'cleanliness': 'czystość', 'clogged': 'zatkany', 'coffee': 'kawa', 'cold': 'zimny', 'comfortable': 'wygodny',
        'compared': 'porównaniu', 'conclusion': 'wniosek', 'conditioning': 'klimatyzacja', 'costing': 'koszt',
        'curtains': 'zasłony', 'customer': 'klient', 'dark': 'ciemny', 'decor': 'wystrój', 'desk': 'biurko',
        'dirty': 'brudny', 'dishes': 'naczynia', 'door': 'drzwi', 'doors': 'drzwi', 'eggs': 'jajka',
        'elevator': 'winda', 'employee': 'pracownik', 'equipment': 'wyposażenie', 'especially': 'szczególnie',
        'excessive': 'wygórowany', 'expensive': 'drogi', 'fan': 'wiatrak', 'floor': 'podłoga', 'flush': 'spłuczka',
        'food': 'jedzenie', 'friendly': 'przyjazny', 'full': 'pełny', 'furnishings': 'meble', 'furniture': 'meble',
        'garden': 'ogród', 'great': 'świetny', 'grumpy': 'zrzędliwy', 'hairdryer': 'suszarka', 'half': 'połowa',
        'hard': 'twardy', 'head': 'głowa', 'heating': 'ogrzewanie', 'hefty': 'wygórowana', 'high': 'wysoki',
        'hot': 'gorący', 'inadequate': 'nieadekwatny', 'including': 'wliczając', 'increase': 'wzrost',
        'informed': 'poinformowany', 'lady': 'pani', 'large': 'duży', 'leaky': 'nieszczelny', 'list': 'lista',
        'location': 'lokalizacja', 'loud': 'głośny', 'lower': 'niższy', 'manager': 'menedżer', 'mattress': 'materac',
        'mattresses': 'materace', 'music': 'muzyka', 'need': 'potrzeba', 'needs': 'potrzeby', 'noise': 'hałas',
        'old': 'stary', 'options': 'opcje', 'outside': 'zewnątrz', 'overall': 'ogólnie', 'paid': 'zapłacony',
        'parking': 'parking', 'pay': 'płacić', 'peeling': 'łuszczący', 'personnel': 'personel', 'pillow': 'poduszka',
        'pillows': 'poduszki', 'pool': 'basen', 'poor': 'słaby', 'price': 'cena', 'prices': 'ceny',
        'quality': 'jakość', 'ratio': 'stosunek', 'reaching': 'osiągający', 'reception': 'recepcja',
        'receptionist': 'recepcjonista', 'receptionists': 'recepcjoniści', 'refrigerator': 'lodówka',
        'restaurant': 'restauracja', 'rooms': 'pokoje', 'rude': 'niegrzeczny', 'sauna': 'sauna',
        'scrambled': 'jajecznica', 'seemed': 'wydawało', 'served': 'obsłużony', 'service': 'obsługa',
        'services': 'usługi', 'sheets': 'prześcieradła', 'shower': 'prysznic', 'significant': 'znaczący',
        'single': 'pojedynczy', 'sink': 'umywalka', 'small': 'mały', 'smell': 'zapach', 'soap': 'mydło',
        'sofa': 'sofa', 'soft': 'miękki', 'something': 'coś', 'sound': 'dźwięk', 'spa': 'spa', 'space': 'przestrzeń',
        'staff': 'personel', 'stained': 'poplamiony', 'street': 'ulica', 'tea': 'herbata', 'temperature': 'temperatura',
        'terrible': 'okropny', 'thin': 'cienki', 'toilet': 'toaleta', 'torn': 'podarty', 'towel': 'ręcznik',
        'towels': 'ręczniki', 'tracks': 'tory', 'trainee': 'stażysta', 'tram': 'tramwaj', 'tv': 'tv', 
        'uncomfortable': 'niewygodny', 'unfortunate': 'niefortunny', 'unfriendly': 'nieprzyjazny',
        'unpleasant': 'nieprzyjemny', 'usable': 'użyteczny', 'value': 'jakość', 'ventilation': 'wentylacja',
        'view': 'widok', 'waiter': 'kelner', 'wallpaper': 'tapeta', 'walls': 'ściany', 'water': 'woda',
        'wellness': 'wellness', 'window': 'okno', 'windows': 'okna', 'worn': 'zużyty'
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, category in enumerate(categories):
        subset = df_neg_full[df_neg_full["aspect_category"] == category]
        
        category_bigrams = []
        for _, row in subset.iterrows():
            text = str(row["neg_text"]).lower()
            aspect = str(row["extracted_aspect"]).lower()
            
            words = re.findall(r'\b\w+\b', text)
            
            for j in range(len(words)-1):
                w1, w2 = words[j], words[j+1]
                # Look for collocations tied directly to the aspect noun
                if w1 == aspect or w2 == aspect:
                    if w1 not in boring_words and w2 not in boring_words:
                        t1 = PL_TRANSLATION_DICT.get(w1, w1)
                        t2 = PL_TRANSLATION_DICT.get(w2, w2)
                        # Join with dash so WordCloud treats it as a single chunk
                        category_bigrams.append(f"{t1}-{t2}")
                        
        text_for_cloud = " ".join(category_bigrams)
        ax = axes[i]
        
        if text_for_cloud.strip():
            # colormap="Dark2" zlikwiduje białe, żółte i puste kolory, faworyzując ostre i widoczne ciemne
            # Bardziej płaski, rozszerzony w poziomie format (2:1) oszczędzający miejsce w Wordzie, ale rozległy (1000px) by dbać o widoczność
            wordcloud = WordCloud(width=1000, height=500, background_color="white",
                                  colormap="Dark2", max_words=30, min_font_size=26, 
                                  collocations=False).generate(text_for_cloud)
            ax.imshow(wordcloud, interpolation='bilinear')
        else:
            ax.text(0.5, 0.5, 'Brak Danych (brak kolokacji)', horizontalalignment='center', verticalalignment='center', fontsize=16)
            
        ax.axis("off")
        ax.set_title(f"Kontekst (N-gramy): {cat_titles[category]}", fontsize=24, pad=15, fontweight='bold')
        
    plt.tight_layout(pad=4.0)
    plt.savefig("data/04_results/figures/fig_06_absa_wordcloud_grid_3x2.png", dpi=300, bbox_inches='tight')
    plt.close()
        
    print("[✓] Fully grouped 3x2 grid wordclouds saved (all-in-one).")

    # -------------------------------------------------------------------------
    # PLOT 07: Heatmap (Hotel Stars vs. Complaint Category)
    # -------------------------------------------------------------------------
    # Calculate negative proportion per specific hotel star rating category
    cross_tab = pd.crosstab(df_neg["hotel_stars"], df_neg["kategoria_pl"], normalize='index') * 100
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="OrRd", linewidths=.5, cbar_kws={'label': 'Odsetek opinii (%)'})
    plt.title("Struktura problematyki w zależności od standardu hotelu (%)", fontsize=14, pad=15)
    plt.xlabel("Kategoria Skargi", fontsize=12)
    plt.ylabel("Standard Hotelu (Liczba Gwiazdek)", fontsize=12)
    
    # Fix Y-axis string representation for Polish audience
    plt.yticks(rotation=0)
    plt.gca().set_yticklabels([f"{int(y)} Gwiazdki" for y in cross_tab.index])
    
    plt.tight_layout()
    plt.savefig("data/04_results/figures/fig_07_absa_sentiment_heatmap.png", dpi=300)
    plt.close()
    print("[✓] fig_07_absa_sentiment_heatmap.png saved.")

    print("=" * 60)
    print("PB4 visualizations complete. Module finished.")

if __name__ == "__main__":
    generate_outputs()
