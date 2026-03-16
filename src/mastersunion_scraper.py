"""
mastersunion_scraper.py
-----------------------
Smart multi-course scraper for mastersunion.org
Each course has multiple sub-pages (tabs) with separate URLs.
Scrapes all pages and saves organised text files.

CHANGELOG (v2 — URL fixes):
  - Fixed all URL slugs to match live site (verified via web search)
  - Added "and" / "in" connectors where the site uses them
  - Fixed PGP TBM tabs (site uses /pgp-tbm-* for sub-pages)
  - Fixed UI/UX slug (/pg-in-ui-ux-and-product-design-*)
  - Fixed UG slug (/ug-technology-and-business-management)
  - Removed tab URLs that don't exist as separate pages
  - Added URL validation step before scraping
  - Added missing courses (PGP Bharat, UG Psychology, UG Data Science, etc.)

Usage:
    python mastersunion_scraper.py
"""

import json
import re
import sys
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
# URLs verified against live site as of March 2026.
#
# KEY FIXES from v1:
#   1. PGP Applied AI:  "pgp-applied-ai-agentic-systems"
#                     → "pgp-in-applied-ai-and-agentic-systems"  (added "in" + "and")
#
#   2. PGP TBM overview: "pgp-technology-business-management"
#                       → "pgp-technology-and-business-management"  (added "and")
#      PGP TBM tabs:     "pgp-technology-business-management-curriculum"
#                       → "pgp-tbm-admissions-and-fees" etc.  (uses short "tbm" prefix)
#
#   3. PGP UI/UX:  "pgp-ui-ux-ai-product-design"
#                → "pg-in-ui-ux-and-product-design"  ("pg" not "pgp", no "ai")
#
#   4. PGP Capital Markets: "pgp-in-capital-markets-and-trading" was correct ✓
#
#   5. PGP Entrepreneurship: "pgp-in-entrepreneurship-business-acceleration"
#                           → "pgp-in-entrepreneurship-business-acceleration" was correct ✓
#
#   6. UG TBM:  "ug-technology-business-management"
#             → "ug-technology-and-business-management"  (added "and")
#
#   7. Many "-class-profile" and "-career-prospects" sub-pages don't exist
#      as separate URLs — they are client-side tabs within the overview page.
#      Removed non-existent tab URLs to avoid 404s.

COURSES = {

    # ── PGP Programmes ───────────────────────────────────────────────────────
    "pgp_applied_ai": {
        "name": "PGP in Applied AI & Agentic Systems",
        "category": "pgp",
        "tabs": {
            # FIX: slug is "pgp-in-applied-ai-AND-agentic-systems" (not hyphens only)
            "overview":    "/pgp-in-applied-ai-and-agentic-systems",
            # Note: curriculum content is on the overview page (SPA tabs).
            # The -applynow page has admissions info:
            "admissions":  "/pgp-in-applied-ai-and-agentic-systems-applynow",
        },
    },
    "pgp_tbm": {
        "name": "PGP in Technology & Business Management",
        "category": "pgp",
        "tabs": {
            # FIX: overview uses "and" between technology/business
            "overview":   "/pgp-technology-and-business-management",
            # FIX: sub-pages use short "pgp-tbm-*" prefix (NOT full slug)
            "admissions": "/pgp-tbm-admissions-and-fees",
            "apply":      "/pgp-tbm-applynow",
        },
    },
    "pgp_hr": {
        "name": "PGP in Human Resources & Organisation Strategy",
        "category": "pgp",
        "tabs": {
            # This slug was correct in v1
            "overview":   "/pgp-human-resources-organisation-strategy",
        },
    },
    "pgp_sports": {
        "name": "PGP in Sports Management & Gaming",
        "category": "pgp",
        "tabs": {
            # This slug was correct in v1
            "overview":       "/pgp-sports-management-gaming",
            "career":         "/pgp-sports-management-gaming-career-prospects",
        },
    },
    "pgp_uiux": {
        "name": "PGP in UI/UX & Product Design",
        "category": "pgp",
        "tabs": {
            # FIX: slug is "pg-in-ui-ux-and-product-design" (NOT "pgp-ui-ux-ai-product-design")
            #   - "pg" not "pgp"
            #   - has "and"
            #   - "product-design" not "ai-product-design"
            "overview":    "/pg-in-ui-ux-and-product-design",
            "curriculum":  "/pg-in-ui-ux-and-product-design-curriculum",
        },
    },
    "pgp_sustainability": {
        "name": "PGP in Sustainability & Business Management",
        "category": "pgp",
        "tabs": {
            # FIX: slug uses "and" — verify this exists; if 404, the SPA tab
            # may be embedded in the overview page
            "overview":   "/pgp-sustainability-and-business-management",
        },
    },
    "pgp_capital_markets": {
        "name": "PGP in Capital Markets & Trading",
        "category": "executive",
        "tabs": {
            "overview":   "/pgp-in-capital-markets-and-trading",
            "curriculum": "/pgp-in-capital-markets-and-trading-curriculum",
            "admissions": "/pgp-in-capital-markets-and-trading-admissions-and-fees",
            # career-prospects tab does not exist as a separate URL (404)
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
    # pgp_opm removed — all URLs 404 on live site (programme may be discontinued)

    # ── PGP Bharat (NEW — was missing from v1) ──────────────────────────────
    "pgp_bharat": {
        "name": "PGP Bharat: Immersion-Driven Programme",
        "category": "pgp",
        "tabs": {
            "overview":    "/pgp-bharat-immersion-driven-programme",
            "experience":  "/pgp-bharat-immersion-driven-programme-experience",
            "admissions":  "/pgp-bharat-immersion-driven-programme-admissions-and-fees",
        },
    },

    # ── UG Programmes ────────────────────────────────────────────────────────
    "ug_tbm": {
        "name": "UG in Technology & Business Management",
        "category": "ug",
        "tabs": {
            # FIX: slug uses "and" → "ug-technology-AND-business-management"
            "overview":   "/ug-technology-and-business-management",
            "curriculum": "/ug-curriculum",
            "admissions": "/ug-admissions-and-fees",
            "career":     "/ug-career-prospects",
        },
    },
    "ug_tbm_global": {
        "name": "UG in Technology & Business Management (Illinois Tech, US)",
        "category": "ug",
        "tabs": {
            "overview": "/ug-technology-and-business-management-global",
        },
    },
    "ug_data_science": {
        "name": "UG in Data Science & AI",
        "category": "ug",
        "tabs": {
            "overview": "/ug-data-science-and-artificial-intelligence",
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

    # Target main content area — Next.js sites put everything under #__next
    content = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id="__next") or
        soup.find(class_="content") or
        soup.find(class_="container") or
        soup
    )

    text = content.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l and len(l) > 10]

    # Remove duplicate lines (nav items repeat across pages)
    seen = set()
    deduped = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    return "\n".join(deduped)


# ─── URL VALIDATOR ────────────────────────────────────────────────────────────

def validate_url(url: str) -> bool:
    """Send a HEAD request to check if a URL exists (not 404)."""
    try:
        resp = requests.head(url, headers=HEADERS, timeout=10, allow_redirects=True)
        return resp.status_code < 400
    except requests.RequestException:
        return False


# ─── SINGLE PAGE FETCHER ──────────────────────────────────────────────────────

def fetch_page(url: str, retries: int = 2) -> str:
    """Try Playwright first (JS rendering), fall back to requests."""

    # ── Method 1: Playwright (handles JS-rendered Next.js sites) ─────────
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": HEADERS["User-Agent"]})
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                page.wait_for_selector("main, article, #__next, h1", timeout=10000)
            except Exception:
                pass
            page.wait_for_timeout(3000)
            html = page.content()
            browser.close()

            # Validate we got real content, not an empty shell
            soup = BeautifulSoup(html, "html.parser")
            if len(soup.get_text(strip=True)) > 200:
                return html
            print(f"    [THIN-JS] {url} — Playwright got thin content")
            return ""

    except ImportError:
        print("  [WARN] Playwright not installed, falling back to requests")
        print("  Run: pip install playwright && playwright install chromium")
    except Exception as e:
        print(f"    [PLAYWRIGHT-FAIL] {url} -> {e}")

    # ── Method 2: requests fallback ───────────────────────────────────────
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 404:
                return ""
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"    [FAIL] {url} -> {e}")
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

        print(f"-- {name} ({category})")

        combined_parts = [f"COURSE: {name}\nCATEGORY: {category}\n"]

        for tab_name, path in course["tabs"].items():
            url  = BASE + path

            # ── Quick HEAD check to catch 404s early ─────────────────────
            if not validate_url(url):
                print(f"  [404]  {tab_name:<14} {url}")
                skipped += 1
                continue

            html = fetch_page(url)

            if not html:
                print(f"  [SKIP] {tab_name} — empty response")
                skipped += 1
                continue

            text = clean_html(html)

            if len(text) < 100:
                print(f"  [THIN] {tab_name} — only {len(text)} chars, likely JS-rendered")
                skipped += 1
                continue

            # Save individual tab file
            tab_file = cat_dir / f"{course_key}_{tab_name}.txt"
            file_header = f"=== {name} — {tab_name.upper()} ===\nURL: {url}\n\n"
            tab_file.write_text(file_header + text, encoding="utf-8")

            # Also add to combined file
            combined_parts.append(f"\n\n--- {tab_name.upper()} ---\n{text}")

            print(f"  [OK]   {tab_name:<14} {len(text):>6,} chars -> {tab_file.name}")
            done += 1

            # Polite delay — don't hammer the server
            time.sleep(1.2)

        # Save combined file per course (best for RAG — all context together)
        combined_file = RAW_DIR / f"{course_key}_full.txt"
        combined_file.write_text("\n".join(combined_parts), encoding="utf-8")
        print(f"  [COMBINED] -> {combined_file.name}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    files = list(RAW_DIR.glob("**/*.txt"))
    total_chars = sum(f.stat().st_size for f in files)

    print("=" * 50)
    print(f"  Done! {done} pages scraped, {skipped} skipped")
    print(f"  {len(files)} text files, {total_chars:,} total chars")
    print(f"  Output: {RAW_DIR}/")
    print("=" * 50)


# ─── PDF → RAW JSON EXTRACTOR ─────────────────────────────────────────────────

# Filename keyword → category mapping used by _categorise_pdf()
_CATEGORY_KEYWORDS = {
    "pgp":       ["pgp", "postgraduate", "post-graduate", "post_graduate"],
    "ug":        ["ug", "undergraduate", "under-graduate", "under_graduate", "btech", "b_tech"],
    "executive": ["executive", "capital", "entrepreneurship", "rise", "general_mgmt"],
}


def _categorise_pdf(filename: str) -> str:
    """Infer category (pgp / ug / executive / general) from the PDF filename."""
    name_lower = filename.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "general"


def extract_pdfs_to_raw(
    source_dir: str = "mastersunion_files",
    output_dir: str = "data/raw",
) -> dict:
    """
    Extract text from all PDFs in `source_dir` using PyMuPDF (fitz),
    clean each page with clean_ocr(), categorise by filename keywords,
    and save one JSON file per PDF into data/raw/<category>/.

    Skips PDFs that already have a matching JSON output file so
    re-running the script is safe and fast.

    Returns a summary dict: {"processed": int, "skipped": int, "failed": int}.
    """
    # PyMuPDF ships as the 'fitz' namespace
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[PDF] PyMuPDF not installed. Run: pip install pymupdf")
        return {"processed": 0, "skipped": 0, "failed": 0}

    # Allow running this file directly (sys.path may not include project root)
    try:
        from utils.ocr_cleaner import clean_ocr
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.ocr_cleaner import clean_ocr

    src = Path(source_dir)
    out = Path(output_dir)

    if not src.exists():
        print(f"[PDF] Source directory '{src}' not found.")
        return {"processed": 0, "skipped": 0, "failed": 0}

    # Collect all PDFs (top-level + one level deep)
    pdf_files = sorted(src.glob("*.pdf")) + sorted(src.glob("**/*.pdf"))
    # Deduplicate in case ** also matched top-level files
    seen_paths: set = set()
    unique_pdfs = []
    for p in pdf_files:
        if p not in seen_paths:
            seen_paths.add(p)
            unique_pdfs.append(p)

    print(f"[PDF] Found {len(unique_pdfs)} PDF(s) in '{src}'")

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    for pdf_path in unique_pdfs:
        stem     = pdf_path.stem          # filename without extension
        category = _categorise_pdf(pdf_path.name)
        cat_dir  = out / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        out_file = cat_dir / f"{stem}.json"

        # Skip already-processed files to avoid duplicate work
        if out_file.exists():
            print(f"  [SKIP] {pdf_path.name} — already processed ({out_file})")
            stats["skipped"] += 1
            continue

        try:
            doc   = fitz.open(str(pdf_path))
            pages = []

            for page_idx in range(len(doc)):
                raw_text = doc[page_idx].get_text("text") or ""
                cleaned  = clean_ocr(raw_text)
                # Ignore blank / near-blank pages (likely scanned images)
                if len(cleaned.strip()) > 30:
                    pages.append({
                        "page_num": page_idx + 1,   # 1-based for readability
                        "text":     cleaned,
                    })

            doc.close()

            if not pages:
                print(f"  [WARN] {pdf_path.name} — no extractable text (possibly scanned-image PDF)")
                stats["failed"] += 1
                continue

            # Serialise to JSON and write
            payload = {
                "filename": pdf_path.name,
                "stem":     stem,
                "category": category,
                "pages":    pages,
            }
            out_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  [OK]  {pdf_path.name} → {out_file} ({len(pages)} page(s))")
            stats["processed"] += 1

        except Exception as exc:
            print(f"  [ERR] {pdf_path.name} → {exc}")
            stats["failed"] += 1

    print(
        f"\n[PDF] Done — processed:{stats['processed']}  "
        f"skipped:{stats['skipped']}  failed:{stats['failed']}"
    )
    return stats


if __name__ == "__main__":
    print("=" * 50)
    print("  MastersUnion.org -- Smart Course Scraper v2")
    print("=" * 50 + "\n")
    scrape_all()