import html
import smtplib
import json
import os
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

FLAGGED_PATH = "data/flagged_queries.json"
_lock = threading.Lock()


def _save_to_json(query: str, chat_history: list, answer_attempt: str) -> None:
    """Always save flagged query locally so admin can review even if email fails."""
    history_snippet = [{"role": r, "msg": m} for r, m in chat_history[-4:]] if chat_history else []
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "query": query,
        "chat_history": history_snippet,
        "answer_attempt": answer_attempt[:500],
        "resolved": False,
    }

    with _lock:
        existing = []
        if os.path.exists(FLAGGED_PATH):
            try:
                with open(FLAGGED_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(entry)
        os.makedirs("data", exist_ok=True)
        with open(FLAGGED_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"[FLAGGED] Saved to {FLAGGED_PATH}: {query[:60]}")


def notify_admin(query: str, chat_history: list, answer_attempt: str) -> None:
    """Save flagged query to JSON and email admin.

    Gmail setup (required):
      1. Enable 2-Step Verification on the Gmail account
      2. Go to myaccount.google.com → Security → App Passwords
      3. Generate a 16-char App Password (looks like: abcd efgh ijkl mnop)
      4. Put that code (without spaces) in secrets.toml as SMTP_PASSWORD

    secrets.toml:
        SMTP_USER     = "your-gmail@gmail.com"
        SMTP_PASSWORD = "abcdefghijklmnop"   # 16-char App Password
        ADMIN_EMAIL   = "pgadmissions@mastersunion.org"
    """
    # Always save locally first — email is best-effort
    _save_to_json(query, chat_history, answer_attempt)

    try:
        import streamlit as st
        smtp_user   = st.secrets.get("SMTP_USER")    or os.environ.get("SMTP_USER", "")
        smtp_pass   = st.secrets.get("SMTP_PASSWORD") or os.environ.get("SMTP_PASSWORD", "")
        admin_email = st.secrets.get("ADMIN_EMAIL")  or os.environ.get("ADMIN_EMAIL", "")
    except Exception:
        smtp_user   = os.environ.get("SMTP_USER", "")
        smtp_pass   = os.environ.get("SMTP_PASSWORD", "")
        admin_email = os.environ.get("ADMIN_EMAIL", "")

    if not all([smtp_user, smtp_pass, admin_email]):
        print("[EMAIL] Skipped: SMTP_USER / SMTP_PASSWORD / ADMIN_EMAIL not set")
        return

    history_text = (
        "\n".join(f"{r}: {m}" for r, m in chat_history[-4:])
        if chat_history else "(No prior history)"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[PGP Bot] Unhandled query — {datetime.now().strftime('%d %b %Y %H:%M')}"
    msg["From"]    = smtp_user
    msg["To"]      = admin_email

    safe_query   = html.escape(query)
    safe_history = html.escape(history_text)
    safe_answer  = html.escape(answer_attempt[:800])
    html_body = f"""
    <html><body style="font-family:sans-serif;line-height:1.6;">
    <h3 style="color:#c0392b;">&#9888; Unanswered Query Alert</h3>
    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M IST')}</p>
    <p><strong>Query:</strong> {safe_query}</p>
    <hr>
    <p><strong>Recent chat history:</strong></p>
    <pre style="background:#f4f4f4;padding:10px;border-radius:4px;font-size:13px;">{safe_history}</pre>
    <hr>
    <p><strong>Bot's attempted answer:</strong></p>
    <pre style="background:#f4f4f4;padding:10px;border-radius:4px;font-size:13px;">{safe_answer}</pre>
    <hr>
    <p style="color:#888;font-size:12px;">
      Review &amp; add answers at the Admin Panel page in the app sidebar.<br>
      Or manually append to <code>data/program_data.txt</code> and restart.
    </p>
    </body></html>
    """
    msg.attach(MIMEText(html_body, "html"))

    # Try STARTTLS port 587 first (required for Gmail App Passwords)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, admin_email, msg.as_string())
        print(f"[EMAIL] Sent via STARTTLS for query: {query[:60]}")
        return
    except Exception as e1:
        print(f"[EMAIL] STARTTLS failed: {e1}")

    # Fallback: SSL port 465
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, admin_email, msg.as_string())
        print(f"[EMAIL] Sent via SSL for query: {query[:60]}")
    except Exception as e2:
        print(f"[EMAIL] SSL also failed: {e2}")
        print("[EMAIL] Check that SMTP_PASSWORD is a Gmail App Password, not your login password.")
        print("[EMAIL] Get one at: myaccount.google.com → Security → App Passwords")
