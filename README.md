# 🏨 Booking.com Hotel Sentiment Engine & Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=for-the-badge&logo=pandas&logoColor=white)
![Playwright](https://img.shields.io/badge/Playwright-Scraping-red?style=for-the-badge&logo=playwright&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-DeBERTa%20%7C%20ABSA-orange?style=for-the-badge)
![Stats](https://img.shields.io/badge/Stats-Kruskal--Wallis%20%7C%20Dunn-purple?style=for-the-badge)

A complete end-to-end data engineering, statistical, and Natural Language Processing (NLP) pipeline designed to extract, process, and evaluate guest satisfaction across the hospitality industry. The project acts as the technical groundwork for a comprehensive academic thesis evaluating guest satisfaction using Wrocław (Poland) as a case study.

## 📌 Project Overview
The primary goal of this repository is to analyze the gap between hotel star ratings, traveler profiles, and real-world guest feedback. It processes raw review data into actionable management insights using **Non-Parametric Statistical Testing** and **Aspect-Based Sentiment Analysis (ABSA)**.

### Research Problems Addressed (PB):
* **PB1:** How does the star category impact overall guest satisfaction? *(Kruskal-Wallis H Test & Dunn's Post-Hoc)*
* **PB2:** Does the traveler's profile (Business, Family, Solo, etc.) dictate a harsher scoring mechanism? *(Kruskal-Wallis H Test & Dunn's Post-Hoc)*
* **PB3:** Are domestic (Polish) tourists grading hotels differently than international visitors? *(Mann-Whitney U Test)*
* **PB4 (NLP):** What are the hidden, categorical roots of complaints hidden inside text reviews, and how do they map to hotel standards? *(PyABSA DeBERTa Model & Custom N-gram Collocations)*

---

## 📊 Demographic & Market Findings

In addition to formal hypothesis testing, the pipeline evaluates the overall market structure through direct quantitative indicators.

* **Language Structure:** The majority of analyzed reviews are domestic (Polish), followed closely by English, German, and a statistically notable Czech representation.
* **Negative Rate Indicator:** A macro-economic view showing the direct percentage of severe complaints (`negative_total` / `total reviews`) mapped against the hotel's star rating. Demonstrates a clear operational divide between 3★ and 5★ tiers.

---

## ⚙️ Architecture & Pipeline

### 1. Data Collection (`01_collect_data.py`)
A highly resilient, custom asynchronous scraper built on **Playwright**. Designed to bypass complex anti-bot protection mechanisms.
* **Target:** 20 specifically selected hotels across Wrocław (3★, 4★, and 5★ tiers).
* **Dataset Size:** ~6,700 unique reviews preserved after deduplication.

### 2. Preprocessing (`02a_preprocess.py` & `02b_recover_business_travelers.py`)
Cleans and structures the data into dual datasets, alongside a specific sub-routine to recover shadow-banned 'Business' profile types stripped by the booking platform.
* **`reviews_stats.csv`**: Aggregated metadata used for PB1-PB3 statistical tests.
* **`absa_input.csv`**: A language-filtered dataset dedicated exclusively to the PB4 NLP model.

### 3. Statistical Validation (`04_pb1_pb3_analyse_statistics.py`)
Automated generation of robust, academic-grade reports. Checks distributions via **Shapiro-Wilk** and homoscedasticity via **Levene's test** before seamlessly applying Non-Parametric alternatives (due to non-normal skew typical in e-commerce reviews).

### 4. Advanced NLP & ABSA (`05_pb4_absa_inference.py` & `06_pb4_absa_visualize.py`)
Utilizes a locally-adapted PyABSA Aspect-Based approach to extract raw complaint targets, and later groups them into actionable business categories (e.g., STAFF, CLEANLINESS, COMFORT). 
* **Expanded Dictionary:** Custom mapping dict dynamically expanded based on zero-shot fallbacks.
* **Pseudo-OTE N-Grams:** Custom bigram extractor bridging aspects with adjoining adjectives (e.g., `dirty-shower`) visualized through Dark-mode wordclouds.

---

## 🛠 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dominikgabrych/hotel-sentiment-engine.git
   cd hotel-sentiment-engine
   ```

2. **Setup virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Playwright Browsers (for Scraper):**
   ```bash
   playwright install chromium
   ```

## 🚀 The Numerical Execution Pipeline

The repository operates on a strictly chronological structure (01 to 07). To replicate the entire thesis findings from scratch, execute the scripts sequentially:

```bash
# 1. Scrape the data
python analysis_scripts/01_collect_data.py --pass1 --pass2

# 2. Clean, filter, and recover missing profiles
python analysis_scripts/02a_preprocess.py
python analysis_scripts/02b_recover_business_travelers.py

# 3. Generate Baseline Market Demographics (PB0)
python analysis_scripts/03_pb0_distribution_visualize.py

# 4. Generate PB1, PB2, and PB3 statistics and distributions
python analysis_scripts/04_pb1_pb3_analyse_statistics.py

# 5. Extract specific complaints using PyTorch DeBERTa (ABSA Inference)
python analysis_scripts/05_pb4_absa_inference.py

# 6. Group ABSA findings and generate high-level heatmaps and wordclouds
python analysis_scripts/06_pb4_absa_visualize.py

# 7. Generate formatted Markdown tables summarizing the entire project
python analysis_scripts/07_generate_word_tables.py
```

---

## 📁 Repository Structure
```text
.
├── analysis_scripts/            # Core chronological scripts (The Numerical Pipeline 01-07)
├── booking/                     # Scraper class & logic definitions
├── data/
│   ├── 01_raw/                  # Direct Playwright scraper outputs
│   ├── 02_cleaned/              # Structured statistical dataset
│   ├── 03_processed/            # NLP-ready dataset
│   └── 04_results/              # Final deliverables (figures & markdown tables)
├── requirements.txt             # Minimal project dependencies
└── README.md                    # Project documentation
```