import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

ARTIFACT_DIR = "/home/domin/.gemini/antigravity-ide/brain/ac15bc34-8f64-49b6-bfcb-b4d008bb6b96/"
OUT_DIR = "data/04_results/figures/"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# Load the file
df = pd.read_csv("data/01_raw/review_distribution.csv")

# 1. Struktura Językowa (Pie Chart lub Bar Chart)
language_counts = {}
for idx, row in df.iterrows():
    # JSON strings in CSV have double quotes often double escaped, safely evaluate
    try:
        lang_dict = json.loads(row['languages_json'])
    except:
        try:
            lang_dict = ast.literal_eval(row['languages_json'])
        except:
            continue
            
    for lang, count in lang_dict.items():
        language_counts[lang] = language_counts.get(lang, 0) + count

# Wyodrębnienie specyficznych 4 języków na polecenie (Polski, Angielski, Niemiecki, Czeski)
lang_series = pd.Series(language_counts).sort_values(ascending=False)
desired_langs = ['pl', 'en', 'de', 'cs']

final_dict = {l: lang_series.get(l, 0) for l in desired_langs}
other_sum = lang_series[~lang_series.index.isin(desired_langs)].sum()
final_dict['Inne'] = other_sum

final_langs = pd.Series(final_dict).sort_values(ascending=False)
if 'Inne' in final_langs:
    inne_val = final_langs.pop('Inne')
    final_langs['Inne'] = inne_val

# Słownik tłumaczeń dla legendy
lang_translate = {"pl": "Polski", "en": "Angielski", "de": "Niemiecki", "cs": "Czeski"}
final_langs.index = [lang_translate.get(x, x) for x in final_langs.index]

plt.figure(figsize=(9, 6))
colors = sns.color_palette("pastel")[0:6]
explode_settings = (0, 0, 0, 0, 0) # Bez wysunięcia, po prostu zmieniamy rotację tortu
plt.pie(final_langs.values, labels=final_langs.index, autopct='%1.1f%%', startangle=90, 
        colors=colors, wedgeprops={'edgecolor': 'black'}, pctdistance=0.75, explode=explode_settings)
plt.title("Struktura Językowa Zostawianych Opinii", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_00a_language_distribution.png"), dpi=300)
plt.savefig(os.path.join(ARTIFACT_DIR, "fig_00a_language_distribution.png"), dpi=300) # Do podglądu
plt.close()

# 2. Wskaźnik Negatywności (Negative Rate) wg Gwiazdek
# Zamiast robić ranking hoteli, pokażmy odsetek problemów (negative_total / total)
df['negative_rate'] = (df['negative_total'] / df['total']) * 100
negative_by_stars = df.groupby('hotel_stars')['negative_rate'].mean()

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=negative_by_stars.index, y=negative_by_stars.values, palette="Reds_r")
plt.title("Wskaźnik Negatywności (Odsetek Skarg) wg Standardu Hotelu", fontsize=14, fontweight='bold')
plt.xlabel("Standard Hotelu (Gwiazdki)", fontsize=12)
plt.ylabel("Odsetek Skarg (%)", fontsize=12)

# Odsunięcie osi Y, by 10.9% się zmieściło
plt.ylim(0, negative_by_stars.max() * 1.2)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_00e_negative_rate_by_stars.png"), dpi=300)
plt.savefig(os.path.join(ARTIFACT_DIR, "fig_00e_negative_rate_by_stars.png"), dpi=300)
plt.close()

print("Wygenerowano 5 demograficznych wykresów!")
