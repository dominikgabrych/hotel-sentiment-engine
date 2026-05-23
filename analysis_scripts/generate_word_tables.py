import pandas as pd
import os

RESULTS_DIR = "data/04_results/tables/"
OUT_DIR = "data/04_results/word_tables/"

os.makedirs(OUT_DIR, exist_ok=True)

# Słownik tłumaczeń
translation_dict = {
    "Business": "Biznes",
    "Couple": "Pary",
    "Family": "Rodziny",
    "Solo": "Solo",
    "Group of friends": "Grupy znajomych",
    "Other": "Reszta Świata",
    "Poland": "Polska",
    "3": "3 gwiazdki",
    "4": "4 gwiazdki",
    "5": "5 gwiazdek",
    "3 Gwiazdki": "3 gwiazdki",
    "4 Gwiazdki": "4 gwiazdki",
    "5 Gwiazdki": "5 gwiazdek"
}

# -------------------------------------------------------------
# TABELA 1: Charakterystyka
# -------------------------------------------------------------
try:
    t_all_raw = pd.read_csv(RESULTS_DIR + "tab_01a_descriptive_overall.csv")
    t_all = t_all_raw.set_index("Statistic").T.reset_index(drop=True)
    if "Std Dev" in t_all.columns:
        t_all.rename(columns={"Std Dev": "Std"}, inplace=True)
        
    t_all.insert(0, "Podgrupa", "Cała próba")
    t_all.insert(0, "Kategoria", "OGÓŁEM")
    
    t_stars = pd.read_csv(RESULTS_DIR + "tab_01b_descriptive_by_stars.csv")
    t_stars.rename(columns={"hotel_stars": "Podgrupa"}, inplace=True)
    t_stars.insert(0, "Kategoria", "Standard Hotelu")
    t_stars["Podgrupa"] = t_stars["Podgrupa"].apply(lambda x: str(x).replace("★", ""))
    
    t_trav = pd.read_csv(RESULTS_DIR + "tab_01c_descriptive_by_traveler.csv")
    t_trav.rename(columns={"traveler_type": "Podgrupa"}, inplace=True)
    t_trav.insert(0, "Kategoria", "Profil Podróżnego")
    
    t_country = pd.read_csv(RESULTS_DIR + "tab_01d_descriptive_by_country.csv")
    t_country.rename(columns={"origin": "Podgrupa"}, inplace=True)
    t_country.insert(0, "Kategoria", "Pochodzenie")

    t_consolidated = pd.concat([t_all, t_stars, t_trav, t_country], ignore_index=True)
    
    # Zastosowanie polskiego słownika
    t_consolidated["Podgrupa"] = t_consolidated["Podgrupa"].replace(translation_dict)
    
    t_consolidated.rename(columns={
        "N": "Liczba Opinii", "Mean": "Średnia", "Std": "Odch. Stand.", "Median": "Mediana"
    }, inplace=True)
    
    cols_to_keep = ["Kategoria", "Podgrupa", "Liczba Opinii", "Średnia", "Mediana", "Odch. Stand."]
    t_consolidated = t_consolidated[[c for c in cols_to_keep if c in t_consolidated.columns]]
    
    for col in ["Średnia", "Odch. Stand.", "Mediana"]:
        if col in t_consolidated.columns:
            t_consolidated[col] = pd.to_numeric(t_consolidated[col], errors='coerce').round(2)
            
    with open(os.path.join(OUT_DIR, "tabela_1_charakterystyka.md"), "w", encoding="utf-8") as f:
        f.write("## Tabela 1: Charakterystyka statystyczna próby badawczej\n\n")
        f.write(t_consolidated.to_markdown(index=False) + "\n")
except Exception as e:
    print(f"Błąd tabeli 1: {e}")

# -------------------------------------------------------------
# TABELE 2 i 3: Post-Hoc Dunna
# -------------------------------------------------------------
def format_pvalue(x):
    try:
        val = float(x)
        if val >= 0.9999:
            return "1.000"
        elif val < 0.001:
            return "< 0.001"
        else:
            return f"{val:.4f}"
    except:
        return x

try:
    t2 = pd.read_csv(RESULTS_DIR + "tab_02c_pb1_posthoc_dunn.csv", index_col=None)
    t2.columns = [c.replace("★", "") for c in t2.columns]
    
    # Tłumaczenie gwiazdek na słowa
    t2.columns = [translation_dict.get(c, c) for c in t2.columns]
    t2.index = t2.columns
    
    t2 = t2.apply(lambda col: col.map(format_pvalue))
    t2.index.name = "Standard"
    with open(os.path.join(OUT_DIR, "tabela_2_dunna_pb1.md"), "w", encoding="utf-8") as f:
        f.write("## Tabela 2: Macierz p-value testu Post-Hoc Dunna (Dla Gwiazdek Hotelowych - PB1)\n\n")
        f.write(t2.to_markdown() + "\n")
except Exception as e:
    print(f"Błąd tabeli 2: {e}")

try:
    t3 = pd.read_csv(RESULTS_DIR + "tab_03c_pb2_posthoc_dunn.csv", index_col=None)
    
    # Tłumaczenie profili
    t3.columns = [translation_dict.get(c, c) for c in t3.columns]
    t3.index = t3.columns
    
    t3 = t3.apply(lambda col: col.map(format_pvalue))
    t3.index.name = "Profil"
    with open(os.path.join(OUT_DIR, "tabela_3_dunna_pb2.md"), "w", encoding="utf-8") as f:
        f.write("## Tabela 3: Macierz p-value testu Post-Hoc Dunna (Dla Profilu Podróżnego - PB2)\n\n")
        f.write(t3.to_markdown() + "\n")
except Exception as e:
    print(f"Błąd tabeli 3: {e}")

# -------------------------------------------------------------
# TABELA 4: Wyniki ABSA
# -------------------------------------------------------------
try:
    t6 = pd.read_csv(RESULTS_DIR + "tab_06_absa_aspect_counts.csv")
    t6.rename(columns={
        "kategoria_pl": "Kategoria Skargi",
        "NEGATIVE": "Sentyment Negatywny",
        "NEUTRAL": "Sentyment Neutralny",
        "POSITIVE": "Sentyment Pozytywny"
    }, inplace=True)
    
    # Dodanie łącznej sumy na końcu
    if all(col in t6.columns for col in ["Sentyment Negatywny", "Sentyment Neutralny", "Sentyment Pozytywny"]):
        t6["Łączna liczba aspektów"] = t6["Sentyment Negatywny"] + t6["Sentyment Neutralny"] + t6["Sentyment Pozytywny"]
        
    with open(os.path.join(OUT_DIR, "tabela_4_wyniki_absa.md"), "w", encoding="utf-8") as f:
        f.write("## Tabela 4: Główne Kategorie Skarg ze zliczeniem sentymentu (PB4 - Wyniki ABSA)\n\n")
        f.write(t6.to_markdown(index=False) + "\n")
except Exception as e:
    print(f"Błąd tabeli 4: {e}")

print("Wygenerowano zaktualizowane tabele z polskimi nazwami i nowymi kolumnami.")
