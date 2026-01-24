# 🏨 Hotel Sentiment Engine
### AI-Powered Business Intelligence for Hospitality

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![NLP](https://img.shields.io/badge/NLTK-VADER-green)
![Status](https://img.shields.io/badge/Status-Bachelor_Thesis_MVP-orange)

## 📖 Overview
**Hotel Sentiment Engine** is an advanced analytics tool designed to uncover the *true* drivers of guest satisfaction. While traditional platforms rely solely on star ratings (1-10), this project utilizes **Natural Language Processing (NLP)** to decode the emotional tone and specific complaints hidden within written reviews.

Developed as part of a **Bachelor's Thesis**, this project demonstrates how Data Science can solve real-world problems in the hospitality industry by detecting "hidden friction points" that standard metrics miss.

---

## 🧐 The Business Problem
Hotel managers often face a **"Data Blind Spot"**:
1.  **Star Inflation:** Guests often give a 10/10 rating out of politeness, even if they experienced issues (e.g., "Great stay, but the room was cold").
2.  **Volume Overload:** Manually reading thousands of reviews to find patterns is impossible.
3.  **The "Why" Gap:** Knowing *that* the score dropped in May is easy; knowing *why* (e.g., "broken A/C" vs. "staff attitude") requires text analysis.

**This solution bridges the gap between the official score and the actual guest experience.**

---

## 🚀 Key Features

### 1. Robust Data Collection
* **Custom Scraper:** Built with **Playwright**, capable of handling dynamic content and pagination on Booking.com.
* **Ethical Scraping Architecture:** Designed with delays and user-agent rotation to respect server load.

### 2. Advanced NLP Pipeline
* **Sentiment Analysis (VADER):** Calculates a "Text Sentiment Score" (-1 to +1) for every review to detect discrepancies between the stars clicked and words written.
* **Root Cause Analysis:** Uses **Part-of-Speech (POS) Tagging** to filter out generic noise (verbs, filler words) and isolate specific **nouns and adjectives** (e.g., "dirty carpet", "noisy street") that drive negativity.

### 3. Interactive Dashboard
* **Market Segmentation:** Compare performance across different standards (3★ vs 4★ vs 5★).
* **Seasonality Stress Test:** Visualize if quality drops when demand spikes (Occupancy vs. Rating correlation).
* **Demographic Insights:** Analyze rating gaps between local and international guests.
* **Traveler Type:** Identify which groups are the most demanding.

---

## 📸 Dashboard Preview

<img width="2555" height="1047" alt="Zrzut ekranu 2026-01-24 161100" src="https://github.com/user-attachments/assets/d348998f-cf48-4128-80d4-7619009f2abc" />
<img width="2248" height="534" alt="Zrzut ekranu 2026-01-24 161114" src="https://github.com/user-attachments/assets/d00a67a9-9979-4d60-a125-54c187345b17" />

---

## ⚙️ Tech Stack

* **Core:** Python 3.12
* **Data Engineering:** Pandas, NumPy
* **Scraping:** Playwright (Synchronous API)
* **NLP:** NLTK (VADER Lexicon, Averaged Perceptron Tagger)
* **Visualization:** Streamlit, Plotly Express, Plotly Graph Objects

---

## 📂 Dataset & Scale Note
> **⚠️ Demo Sample:** This repository contains a targeted sample dataset of **~450 reviews** from 3 hotels in Wrocław, Poland.

This sample size was chosen intentionally to:
1.  Demonstrate the analytical capabilities without violating scraping policies or overloading external servers.
2.  Focus on specific seasonal anomalies (e.g., holiday peaks).

*The architecture is fully scalable and can process thousands of reviews in production environments.*

---

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/hotel-sentiment-engine.git](https://github.com/your-username/hotel-sentiment-engine.git)
    cd hotel-sentiment-engine
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright browsers (required for scraper):**
    ```bash
    playwright install chromium
    ```

4.  **Run the ETL Pipeline (Optional - if you want to re-process raw data):**
    ```bash
    python process_data.py
    ```

5.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Key Insights from Demo Analysis
Based on the sample data from Wrocław hotels, the engine detected:
* **The "Nuance Gap":** A 0.54 discrepancy between high star ratings and lower text sentiment, indicating guests are more critical in writing than in scoring.
* **Local Market Sensitivity:** Domestic guests rated hotels significantly lower than international tourists, suggesting a price-to-value mismatch for the local currency.
* **Hidden Issues:** Despite high ratings, the NLP engine identified recurring complaints about "heating" and "parking" that were invisible in the aggregate star score.
