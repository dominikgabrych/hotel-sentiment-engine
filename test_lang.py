from playwright.sync_api import sync_playwright

url = "https://www.booking.com/hotel/pl/hotel-piast-wroclaw-centrum.html?lang=en-gb"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    page = context.new_page()
    page.goto(url, wait_until="domcontentloaded")
    
    # Wait for reviews
    try:
        page.wait_for_selector('div[data-testid="review-card"]', timeout=15000)
        cards = page.locator('div[data-testid="review-card"]').all()
        for i in range(min(3, len(cards))):
            card = cards[i]
            pos = card.locator('[data-testid="review-positive-text"]')
            neg = card.locator('[data-testid="review-negative-text"]')
            
            p_text = pos.inner_text() if pos.count() > 0 else ""
            n_text = neg.inner_text() if neg.count() > 0 else ""
            print(f"Review {i}:")
            print(f" Pos: {p_text}")
            print(f" Neg: {n_text}")
            print("---")
    except Exception as e:
        print(e)
    browser.close()
