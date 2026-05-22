"""
pb4_visualize.py — Aspect Category Detection & Visualization (PB4)

ETAP 2: Ostateczne kategoryzowanie, filtrowanie i generowanie polskich wykresów do pracy licencjackiej.
Zgodnie z poleceniem, Agent poddał "OTHER" ekspertyzie i rozszerzył słownik.

Zdefiniowane wykresy:
- fig_05_absa_negative_aspects.png (Bar chart częstotliwości narzekań)
- fig_06_absa_wordcloud_per_aspect.png (6 wordcloudów na jednej siatce)
- fig_07_absa_sentiment_heatmap.png (Gwiazdki vs. Kategoria Narzekań)
- tab_06_absa_aspect_counts.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- SŁOWNIK PO ANALIZIE AGENTA ---
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

# --- TŁUMACZENIA DO WYKRESÓW (FRONT-END PL) ---
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
    """Mapuje frazę do głównej kategorii wykorzystując poszerzony słownik."""
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
    print("ROZPOCZĘCIE GENEROWANIA WIZUALIZACJI PB4 (ABSA)")
    print("=" * 60)
    
    # Utworzenie katalogów
    os.makedirs("data/04_results/tables", exist_ok=True)
    os.makedirs("data/04_results/figures", exist_ok=True)
    
    # 1. Wczytanie i mapowanie
    df = pd.read_csv("data/04_results/absa_results.csv")
    df["aspect_category"] = df["extracted_aspect"].apply(map_aspect_to_category)
    
    # Tłumaczenie na etykiety polskie dla wykresów
    df["kategoria_pl"] = df["aspect_category"].map(POLISH_LABELS)
    
    # 2. Wygenerowanie i zapis tabeli głównej (PB4 tab_06)
    # Tabela policzy wszytko: category, sentiment -> count
    tab_06 = df.groupby(["kategoria_pl", "sentiment"]).size().unstack(fill_value=0).reset_index()
    tab_06.columns.name = None
    tab_06 = tab_06.sort_values(by="NEGATIVE", ascending=False)
    tab_06.to_csv("data/04_results/tables/tab_06_absa_aspect_counts.csv", index=False, encoding="utf-8")
    print("[✓] tab_06_absa_aspect_counts.csv zapisano.")

    # DO WIZUALIZACJI: Filtrujemy tylko NEGATYWY i wyrzucamy 'OTHER'
    df_neg = df[(df["sentiment"] == "NEGATIVE") & (df["aspect_category"] != "OTHER")]
    
    # Styl wykresów (profesjonalny)
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": ".8", "font.size": 11, "font.family": "sans-serif"})

    # -------------------------------------------------------------------------
    # WYKRES 05: Ranking negatywnych aspektów (Bar Chart)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    order = df_neg["kategoria_pl"].value_counts().index
    ax = sns.countplot(y="kategoria_pl", data=df_neg, order=order, palette="rocket")
    
    plt.title("Najczęstsze kategorie skarg gości (Analiza Sentymentu - ABSA)", fontsize=14, pad=15)
    plt.xlabel("Liczba negatywnych wzmianek w opiniach", fontsize=12)
    plt.ylabel("Kategoria Narzekań", fontsize=12)
    
    for container in ax.containers:
        ax.bar_label(container, padding=5)
        
    plt.tight_layout()
    plt.savefig("data/04_results/figures/fig_05_absa_negative_aspects.png", dpi=300)
    plt.close()
    print("[✓] fig_05_absa_negative_aspects.png zapisano.")

    # -------------------------------------------------------------------------
    # WYKRES 06: Wordcloud (6 chmur na jednym obrazku)
    # -------------------------------------------------------------------------
    categories = [cat for cat in BASE_DICTIONARY.keys()]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Ustalamy polskie tytuły zgodnie z nowym słownikiem
    cat_titles = {c: POLISH_LABELS[c] for c in categories}
    
    for i, category in enumerate(categories):
        subset = df_neg[df_neg["aspect_category"] == category]["extracted_aspect"]
        text = " ".join(subset.astype(str).tolist())
        
        ax = axes[i]
        
        if text.strip():
            # Chmura ciemna (biznesowa), max 40 najważniejszych słów
            wordcloud = WordCloud(width=800, height=400, background_color="white",
                                  colormap="inferno", max_words=40).generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
        else:
            ax.text(0.5, 0.5, 'Brak Danych', horizontalalignment='center', verticalalignment='center')
            
        ax.axis("off")
        ax.set_title(f"Słowa Kluczowe: {cat_titles[category]}", fontsize=14, pad=10)
        
    plt.tight_layout(pad=3.0)
    plt.savefig("data/04_results/figures/fig_06_absa_wordcloud_per_aspect.png", dpi=300)
    plt.close()
    print("[✓] fig_06_absa_wordcloud_per_aspect.png zapisano.")

    # -------------------------------------------------------------------------
    # WYKRES 07: Heatmapa (Gwiazdki vs. Kategoria Narzekań)
    # -------------------------------------------------------------------------
    # Sprawdzamy ile % negatywów z hotelu w danej gwiazdce stanowi dana kategoria
    # Heatmap - Normalizacja proporcjonalna (wierszowo wg hoteli/gwiazdek)
    
    cross_tab = pd.crosstab(df_neg["hotel_stars"], df_neg["kategoria_pl"], normalize='index') * 100
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="OrRd", linewidths=.5, cbar_kws={'label': '% wszystkich skarg w danym standardzie'})
    plt.title("Struktura problemów w zależności od standardu hotelu (%)", fontsize=14, pad=15)
    plt.xlabel("Kategoria Skargi", fontsize=12)
    plt.ylabel("Standard Hotelu (Liczba Gwiazdek)", fontsize=12)
    
    # Naprawa indeksu Y
    plt.yticks(rotation=0)
    plt.gca().set_yticklabels([f"{int(y)} Gwiazdki" for y in cross_tab.index])
    
    plt.tight_layout()
    plt.savefig("data/04_results/figures/fig_07_absa_sentiment_heatmap.png", dpi=300)
    plt.close()
    print("[✓] fig_07_absa_sentiment_heatmap.png zapisano.")

    print("=" * 60)
    print("Wizualizacje PB4 gotowe. Moduł zakończony.")

if __name__ == "__main__":
    generate_outputs()
