import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def notify_admin(query: str, chat_history: list, answer_attempt: str) -> None:
    """Email admin when a query can't be answered with confidence.

    Configure via secrets.toml:
        SMTP_USER     = "your-gmail@gmail.com"
        SMTP_PASSWORD = "your-app-password"   # Gmail App Password (not account password)
        ADMIN_EMAIL   = "admin@example.com"   # recipient; can be same as SMTP_USER
    """
    try:
        import streamlit as st
        smtp_user  = st.secrets.get("SMTP_USER")    or os.environ.get("SMTP_USER", "")
        smtp_pass  = st.secrets.get("SMTP_PASSWORD") or os.environ.get("SMTP_PASSWORD", "")
        admin_email = st.secrets.get("ADMIN_EMAIL")  or os.environ.get("ADMIN_EMAIL", "")
    except Exception:
        smtp_user   = os.environ.get("SMTP_USER", "")
        smtp_pass   = os.environ.get("SMTP_PASSWORD", "")
        admin_email = os.environ.get("ADMIN_EMAIL", "")

    if not all([smtp_user, smtp_pass, admin_email]):
        print("[EMAIL] Skipped: SMTP_USER / SMTP_PASSWORD / ADMIN_EMAIL not set in secrets.toml")
        return

    history_text = (
        "\n".join(f"{role}: {msg}" for role, msg in chat_history[-4:])
        if chat_history else "(No prior history)"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[PGP Bot] Unhandled query — {datetime.now().strftime('%d %b %Y %H:%M')}"
    msg["From"]    = smtp_user
    msg["To"]      = admin_email

    html = f"""
    <html><body style="font-family:sans-serif;line-height:1.6;">
    <h3 style="color:#c0392b;">&#9888; Unanswered Query Alert</h3>
    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M IST')}</p>
    <p><strong>Query:</strong> {query}</p>
    <hr>
    <p><strong>Recent chat history:</strong></p>
    <pre style="background:#f4f4f4;padding:10px;border-radius:4px;font-size:13px;">{history_text}</pre>
    <hr>
    <p><strong>Bot's attempted answer:</strong></p>
    <pre style="background:#f4f4f4;padding:10px;border-radius:4px;font-size:13px;">{answer_attempt[:800]}</pre>
    <hr>
    <p style="color:#888;font-size:12px;">
      Action: Add a good answer to <code>data/program_data.txt</code>,
      then restart the app to rebuild the knowledge base.
    </p>
    </body></html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, admin_email, msg.as_string())
        print(f"[EMAIL] Escalation sent for query: {query[:60]}")
    except Exception as e:
        print(f"[EMAIL] Failed to send notification: {e}")
