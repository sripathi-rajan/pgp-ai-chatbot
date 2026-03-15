import requests
from bs4 import BeautifulSoup

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

def web_search_fallback(query):
    """
    Real-time web search using Google Serper API.
    Falls back to direct website fetch if API key not set.
    """

    # ── Method 1: Google Serper (most reliable) ───────────────────────────────
    import os
    serper_key = os.environ.get("SERPER_API_KEY", "")
    if serper_key:
        try:
            headers = {
                "X-API-KEY": serper_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": f"Masters Union PGP AI {query}",
                "num": 5
            }
            resp = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("organic", [])
                combined = ""
                for r in results:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    link = r.get("link", "")
                    combined += f"{title}\n{snippet}\n{link}\n\n"
                print(f"[SEARCH] Serper found {len(results)} results")
                return combined.strip()
        except Exception as e:
            print(f"[SEARCH] Serper failed: {e}")

    # ── Method 2: DuckDuckGo (free alternative) ───────────────────────────────
    try:
        if DDGS is None:
            raise ImportError("DDGS not available")
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"Masters Union PGP AI {query}",
                max_results=4
            ))
        if results:
            print(f"[SEARCH] DuckDuckGo found {len(results)} results")
            combined = ""
            for r in results:
                combined += f"{r.get('title','')}\n{r.get('body','')}\n\n"
            return combined.strip()
        else:
            print("[SEARCH] DuckDuckGo returned empty")
    except Exception as e:
        print(f"[SEARCH] DuckDuckGo failed: {e}")

    # ── Method 3: Direct website fetch (no API needed) ────────────────────────
    try:
        page_map = {
            "faculty":    "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",
            "professor":  "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",
            "mentor":     "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",
            "fees":       "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-applynow",
            "admission":  "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-applynow",
            "apply":      "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-applynow",
            "curriculum": "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",
            "career":     "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",
            "placement":  "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",
        }

        target_url = "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems"
        for key, url in page_map.items():
            if key in query.lower():
                target_url = url
                break

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(target_url, headers=headers, timeout=10)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            lines = [
                l.strip() for l in
                soup.get_text(separator="\n", strip=True).split("\n")
                if len(l.strip()) > 30
            ]
            result = "\n".join(lines[:60])
            print(f"[SEARCH] Direct fetch: {len(result)} chars from {target_url}")
            return result

    except Exception as e:
        print(f"[SEARCH] Direct fetch failed: {e}")

    return ""