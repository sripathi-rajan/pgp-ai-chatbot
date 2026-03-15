"""
mastersunion_scraper.py
-----------------------
Smart multi-course scraper for mastersunion.org
Each course has 5 sub-pages (tabs) with separate URLs.
Scrapes all ~85 URLs and saves organised text files.

Usage:
    python src/mastersunion_scraper.py
"""

import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://mastersunion.org"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── ALL COURSES ON MASTERSUNION.ORG ─────────────────────────────────────────
# Pattern: BASE_SLUG + TAB_SUFFIX forms the full URL
# Each course has up to 5 tab-pages

COURSES = {

    # ── PGP Programmes ───────────────────────────────────────────────────────
    "pgp_applied_ai": {
        "name": "PGP in Applied AI & Agentic Systems",
        "category": "pgp",
        "tabs": {
            "overview":    "/pgp-applied-ai-agentic-systems",
            "curriculum":  "/pgp-applied-ai-agentic-systems-curriculum",
            "admissions":  "/pgp-applied-ai-agentic-systems-admissions-and-fees",
            "career":      "/pgp-applied-ai-agentic-systems-career-prospects",
            "class":       "/pgp-applied-ai-agentic-systems-class-profile",
        },
    },
    "pgp_tbm": {
        "name": "PGP in Technology & Business Management",
        "category": "pgp",
        "tabs": {
            "overview":   "/pgp-technology-business-management",
            "curriculum": "/pgp-technology-business-management-curriculum",
            "admissions": "/pgp-technology-business-management-admissions-and-fees",
            "career":     "/pgp-technology-business-management-career-prospects",
            "class":      "/pgp-technology-business-management-class-profile",
        },
    },
    "pgp_hr": {
        "name": "PGP in Human Resources & Organisation Strategy",
        "category": "pgp",
        "tabs": {
            "overview":   "/pgp-human-resources-organisation-strategy",
            "curriculum": "/pgp-human-resources-organisation-strategy-curriculum",
            "admissions": "/pgp-human-resources-organisation-strategy-admissions-and-fees",
            "career":     "/pgp-human-resources-organisation-strategy-career-prospects",
            "class":      "/pgp-human-resources-organisation-strategy-class-profile",
        },
    },
    "pgp_sports": {
        "name": "PGP in Sports Management & Gaming",
        "category": "pgp",
        "tabs": {
            "overview":   "/pgp-sports-management-gaming",
            "curriculum": "/pgp-sports-management-gaming-curriculum",
            "admissions": "/pgp-sports-management-gaming-admissions-and-fees",
            "career":     "/pgp-sports-management-gaming-career-prospects",
            "class":      "/pgp-sports-management-gaming-class-profile",
        },
    },
    "pgp_uiux": {
        "name": "PGP in UI/UX & AI Product Design",
        "category": "pgp",
        "tabs": {
            "overview":   "/pgp-ui-ux-ai-product-design",
            "curriculum": "/pgp-ui-ux-ai-product-design-curriculum",
            "admissions": "/pgp-ui-ux-ai-product-design-admissions-and-fees",
            "career":     "/pgp-ui-ux-ai-product-design-career-prospects",
            "class":      "/pgp-ui-ux-ai-product-design-class-profile",
        },
    },
    "pgp_sustainability": {
        "name": "PGP in Sustainability & Business Management",
        "category": "pgp",
        "tabs": {
            "overview":   "/pgp-sustainability-business-management",
            "curriculum": "/pgp-sustainability-business-management-curriculum",
            "admissions": "/pgp-sustainability-business-management-admissions-and-fees",
            "career":     "/pgp-sustainability-business-management-career-prospects",
            "class":      "/pgp-sustainability-business-management-class-profile",
        },
    },
    "pgp_capital_markets": {
        "name": "PGP in Capital Markets & Trading",
        "category": "executive",
        "tabs": {
            "overview":   "/pgp-in-capital-markets-and-trading",
            "curriculum": "/pgp-in-capital-markets-and-trading-curriculum",
            "admissions": "/pgp-in-capital-markets-and-trading-admissions-and-fees",
            "career":     "/pgp-in-capital-markets-and-trading-career-prospects",
        },
    },
    "pgp_entrepreneurship": {
        "name": "PGP in Entrepreneurship & Business Acceleration",
        "category": "executive",
        "tabs": {
            "overview":   "/pgp-in-entrepreneurship-business-acceleration",
            "curriculum": "/pgp-in-entrepreneurship-business-acceleration-curriculum",
            "admissions": "/pgp-in-entrepreneurship-business-acceleration-admissions-and-fees",
        },
    },
    "pgp_general_mgmt": {
        "name": "PGP Rise: General Management",
        "category": "executive",
        "tabs": {
            "overview":   "/pgp-rise-general-management",
            "curriculum": "/pgp-rise-general-management-curriculum",
            "admissions": "/pgp-rise-general-management-admissions-and-fees",
        },
    },
    "pgp_opm": {
        "name": "PGP Rise: Owners & Promoters Management",
        "category": "family_business",
        "tabs": {
            "overview":   "/pgp-rise-owners-promoters-management",
            "curriculum": "/pgp-rise-owners-promoters-management-curriculum",
            "admissions": "/pgp-rise-owners-promoters-management-admissions-and-fees",
        },
    },

    # ── UG Programmes ────────────────────────────────────────────────────────
    "ug_tbm": {
        "name": "UG in Technology & Business Management",
        "category": "ug",
        "tabs": {
            "overview":   "/ug-technology-business-management",
            "curriculum": "/ug-curriculum",
            "admissions": "/ug-admissions-and-fees",
            "career":     "/ug-career-prospects",
        },
    },
}

# ─── CONTENT CLEANER ──────────────────────────────────────────────────────────

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "svg",
                     "button", "meta", "link"]):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l and len(l) > 2]

    # Remove duplicate lines (nav items repeat across pages)
    seen = set()
    deduped = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    return "\n".join(deduped)


# ─── SINGLE PAGE FETCHER ──────────────────────────────────────────────────────

def fetch_page(url: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            # Use Playwright for better JS rendering
            html = scrape_with_playwright(url)
            if not html:
                raise Exception("Playwright returned empty")
            return html
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"    [FAIL] {url} → {e}")
                return ""
    return ""


# ─── MAIN SCRAPER ─────────────────────────────────────────────────────────────

def scrape_all():
    total_pages = sum(len(c["tabs"]) for c in COURSES.values())
    done = 0
    skipped = 0

    print(f"Scraping {len(COURSES)} courses, ~{total_pages} pages...\n")

    for course_key, course in COURSES.items():
        category = course["category"]
        name     = course["name"]

        # Create category subfolder
        cat_dir = RAW_DIR / category
        cat_dir.mkdir(exist_ok=True)

        print(f"── {name} ({category})")

        combined_parts = [f"COURSE: {name}\nCATEGORY: {category}\n"]

        for tab_name, path in course["tabs"].items():
            url  = BASE + path
            html = fetch_page(url)

            if not html:
                print(f"  [SKIP] {tab_name} — empty or 404")
                skipped += 1
                continue

            text = clean_html(html)

            if len(text) < 100:
                print(f"  [THIN] {tab_name} — only {len(text)} chars, likely JS-rendered")
                skipped += 1
                continue

            # Save individual tab file
            tab_file = cat_dir / f"{course_key}_{tab_name}.txt"
            header = f"=== {name} — {tab_name.upper()} ===\nURL: {url}\n\n"
            tab_file.write_text(header + text, encoding="utf-8")

            # Also add to combined file
            combined_parts.append(f"\n\n--- {tab_name.upper()} ---\n{text}")

            print(f"  [OK] {tab_name:<14} {len(text):>6,} chars → {tab_file.name}")
            done += 1

            # Polite delay — don't hammer the server
            time.sleep(1.2)

        # Save combined file per course (best for RAG — all context together)
        combined_file = RAW_DIR / f"{course_key}_full.txt"
        combined_file.write_text("\n".join(combined_parts), encoding="utf-8")
        print(f"  [COMBINED] → {combined_file.name}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    files = list(RAW_DIR.glob("**/*.txt"))
    total_chars = sum(f.stat().st_size for f in files)

    print("═" * 50)
    print(f"  Done! {done} pages scraped, {skipped} skipped")
    print(f"  {len(files)} text files, {total_chars:,} total chars")
    print(f"  Output: {RAW_DIR}/")
    print()
    print("  If many tabs show [THIN], the site uses JS rendering.")
    print("  Run: python src/playwright_fallback.py  (see below)")
    print("═" * 50)


# ─── PLAYWRIGHT FALLBACK ─────────────────────────────────────────────────────
# Only needed if BeautifulSoup returns thin/empty content
# Install: pip install playwright && playwright install chromium

def scrape_with_playwright(url: str) -> str:
    """Use this as fallback if requests returns JS-rendered empty pages."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()
            page.set_extra_http_headers({"User-Agent": HEADERS["User-Agent"]})
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)   # wait for JS to render
            html = page.content()
            browser.close()
            return html
    except ImportError:
        print("  Playwright not installed. Run: pip install playwright && playwright install chromium")
        return ""


if __name__ == "__main__":
    print("═" * 50)
    print("  MastersUnion.org — Smart Course Scraper")
    print("═" * 50 + "\n")
    scrape_all()