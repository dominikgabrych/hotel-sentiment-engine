# BACHELOR THESIS CONTEXT
**Topic:** "Comparative analysis of customer satisfaction in selected hotels in Wrocław using sentiment analysis of online reviews."

## USER PERSONA & GOALS
I am a Management student (not a Computer Science major).
- **Primary Goal:** Create a solid analytical chapter for my bachelor thesis focused on **business impact** and usefullness for hotel managers. I want to understand data I've collected and i don't what to blindly trust analysis. I want to point out all the anomalies in data (e.g. positive review in negative review section). The project must deliver deep insights, not just surface-level stats.
- **Secondary Goal:** Build a Streamlit dashboard that helps Hotel Managers improve their service.
- **Priority:** Code quality is secondary; **Data Interpretation and Actionable Insights are KING**. 
- **Constraint:** Do not generate random charts. Every visualization must answer a specific business question (e.g., "Why are customers complaining about Hotel X despite high ratings?").

### PROJECT STRUCTURE:
- **booking/**: Main application package.
  - `models.py`: Data models definitions (hotel data classes/structures).
  - `scraper.py`: Logic responsible for downloading data from the website.
- **output/**: Directory for results.
  - `wroclaw_hotels_analysis.csv`: This is raw data from website.
- **Root scripts**:
  - `app.py`: Streamlit data analysis and visualization.
  - `collect_data.py`: Script that starts the scraping process.
  - `process_data.py`: Script for cleaning and analyzing data (Pandas/NumPy).

## INSTRUCTIONS FOR AGENT
1. **Be an Analyst first:** Explain *why* this chart matters for a hotel manager.
2. **One implementation at the time** we have some work to do so don't try to one shot everything but make small changes incrementally.
3. **Data Integrity:** If you see weird data (e.g., price = 0), flag it. Do not hallucinate data to fill gaps.
4. **Verification:** After editing `app.py`, assume I will run `streamlit run app.py` to check the results visually.
5. **Streamlit Focused:** All visualizations should be implemented in `app.py`.