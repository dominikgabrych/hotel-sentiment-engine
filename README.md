# 🏨 Booking.com Hotel Sentiment Engine & Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=for-the-badge&logo=pandas&logoColor=white)
![Playwright](https://img.shields.io/badge/Playwright-Scraping-red?style=for-the-badge&logo=playwright&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-RoBERTa%20%7C%20ABSA-orange?style=for-the-badge)
![Stats](https://img.shields.io/badge/Stats-Kruskal--Wallis%20%7C%20Dunn-purple?style=for-the-badge)

A complete end-to-end data engineering, statistical, and Natural Language Processing (NLP) pipeline designed to extract, process, and evaluate guest satisfaction across the hospitality industry. The project acts as the technical groundwork for a comprehensive academic thesis evaluating guest satisfaction using Wrocław (Poland) as a case study.

## 📌 Project Overview
The primary goal of this repository is to analyze the gap between hotel star ratings, traveler profiles, and real-world guest feedback. It processes raw review data into actionable management insights using **Non-Parametric Statistical Testing** and **Aspect-Based Sentiment Analysis (ABSA)**.

### Research Problems Addressed (PB):
* **PB1:** How does the star category impact overall guest satisfaction? *(Kruskal-Wallis H Test & Dunn's Post-Hoc)*
* **PB2:** Does the traveler's profile (Business, Family, Solo, etc.) dictate a harsher scoring mechanism? *(Kruskal-Wallis H Test & Dunn's Post-Hoc)*
* **PB3:** Are domestic (Polish) tourists grading hotels differently than international visitors? *(Mann-Whitney U Test)*
* **PB4 (NLP):** What are the hidden, categorical roots of complaints hidden inside text reviews, and how do they map to hotel standards? *(RoBERTa Model & Custom N-gram Collocations)*

---

## 📊 Demographic & Market Findings

In addition to formal hypothesis testing, the pipeline evaluates the overall market structure through direct quantitative indicators.

* **Language Structure:** The majority of analyzed reviews are domestic (Polish), followed closely by English, German, and a statistically notable Czech representation.
* **Negative Rate Indicator:** A macro-economic view showing the direct percentage of severe complaints (`negative_total` / `total reviews`) mapped against the hotel's star rating. Demonstrates a clear operational divide between 3★ and 5★ tiers.

---

## ⚙️ Architecture & Pipeline

### 1. Data Collection (`booking/scraper.py`)
A highly resilient, custom asynchronous scraper built on **Playwright**. Designed to bypass complex anti-bot protection mechanisms.
* **Target:** 20 specifically selected hotels across Wrocław (3★, 4★, and 5★ tiers).
* **Dataset Size:** 6,718 unique reviews preserved after deduplication out of ~8,500 scraped records.

### 2. Preprocessing (`analysis_scripts/preprocess.py`)
Cleans and structures the data into dual datasets:
* **`reviews_stats.csv`**: Aggregated metadata used for PB1-PB3 statistical tests.
* **`absa_input.csv`**: A language-filtered (English-only) dataset dedicated exclusively to the PB4 NLP model.

### 3. Statistical Validation (`analysis_scripts/analyse_statistics.py`)
Automated generation of robust, academic-grade reports. Checks distributions via **Shapiro-Wilk** and homoscedasticity via **Levene's test** before seamlessly applying Non-Parametric alternatives (due to non-normal skew typical in e-commerce reviews).

### 4. Advanced NLP & ABSA (`analysis_scripts/pb4_visualize.py`)
Utilizes an advanced Aspect-Based approach to group textual complaints into actionable business categories (e.g., STAFF, CLEANLINESS, COMFORT). 
* **Expanded Dictionary:** Custom mapping dict dynamically expanded based on zero-shot fallbacks.
* **Pseudo-OTE:** Custom N-gram extractor bridging aspects with adjoining adjectives (e.g., `dirty-shower`).

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

## 🚀 Execution Flow

```bash
# 1. Generate Baseline Market Demographics (Languages & Negative Rate)
python analysis_scripts/pb0_distribution_visualize.py

# 2. Scrape the data
python analysis_scripts/collect_data.py --pass1 --pass2

# 3. Clean and build the structured datasets
python analysis_scripts/preprocess.py

# 4. Generate PB1, PB2, and PB3 statistics and visualizations
python analysis_scripts/analyse_statistics.py

# 5. Generate formatted markdown tables for Word reporting
python analysis_scripts/generate_word_tables.py
```

*Note: The NLP RoBERTa model inference runs natively on a separate GPU environment using `absa_input.csv`, outputting `absa_results.csv` which is then visualized via script #4.*

---

## 📁 Repository Structure
```text
.
├── analysis_scripts/            # Core logical scripts (scraping, cleaning, stats, NLP, tables)
├── booking/                     # Scraper class & logic definitions
├── data/
│   ├── 01_raw/                  # Direct Playwright outputs
│   ├── 02_cleaned/              # Structured statistical dataset
│   ├── 03_processed/            # NLP-ready dataset
│   └── 04_results/              # Final deliverables (charts, formatted markdown tables)
├── requirements.txt             # Minimal project dependencies
└── README.md                    # Project documentation
```