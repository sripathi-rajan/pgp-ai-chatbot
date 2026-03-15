import json
import os
import shutil
import streamlit as st

FLAGGED_PATH = "data/flagged_queries.json"
PROGRAM_DATA_PATH = "data/program_data.txt"

st.set_page_config(page_title="Admin Panel", page_icon="🔧")
st.title("🔧 Admin Panel — Knowledge Base Manager")

# ── Password protection ───────────────────────────────────────────────────────
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    correct = st.secrets.get("ADMIN_PASSWORD")
    if not correct:
        st.error("ADMIN_PASSWORD is not configured in secrets.toml. Login is disabled.")
        st.stop()
    pwd = st.text_input("Admin Password", type="password")
    if st.button("Login"):
        if pwd == correct:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# ── Load flagged queries ──────────────────────────────────────────────────────
if not os.path.exists(FLAGGED_PATH):
    flagged = []
else:
    try:
        with open(FLAGGED_PATH, "r", encoding="utf-8") as f:
            flagged = json.load(f)
    except Exception:
        flagged = []

unresolved = [q for q in flagged if not q.get("resolved")]
resolved   = [q for q in flagged if q.get("resolved")]

st.metric("Unanswered queries", len(unresolved))
st.metric("Resolved queries",   len(resolved))

# ── Unanswered queries ────────────────────────────────────────────────────────
st.subheader("❓ Unanswered Queries")

if not unresolved:
    st.success("All queries are resolved — nothing pending!")
else:
    for i, item in enumerate(unresolved):
        with st.expander(f"{item['query'][:90]}   ·   {item['timestamp']}"):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Bot said:** _{item.get('answer_attempt','')[:300]}..._")

            answer = st.text_area(
                "Write the correct answer to add to the knowledge base:",
                key=f"ans_{i}",
                height=120,
                placeholder="Type a clear, factual answer here...",
            )

            if st.button("💾 Save to Knowledge Base", key=f"save_{i}"):
                if answer.strip():
                    entry = f"\n\nQ: {item['query']}\nA: {answer.strip()}\n"
                    with open(PROGRAM_DATA_PATH, "a", encoding="utf-8") as f:
                        f.write(entry)

                    # Mark resolved in JSON
                    for q in flagged:
                        if (q.get("query") == item["query"]
                                and q.get("timestamp") == item["timestamp"]):
                            q["resolved"] = True
                            q["answer_added"] = answer.strip()
                            break

                    with open(FLAGGED_PATH, "w", encoding="utf-8") as f:
                        json.dump(flagged, f, indent=2, ensure_ascii=False)

                    st.success("Saved! Click **Reload Knowledge Base** below to apply.")
                else:
                    st.warning("Please write an answer before saving.")

# ── Reload pipeline ───────────────────────────────────────────────────────────
st.divider()
st.subheader("🔄 Reload Knowledge Base")
st.caption("After saving new answers above, reload so the bot learns them immediately.")

if st.button("Reload Knowledge Base", type="primary", use_container_width=True):
    if os.path.exists("./faiss_index"):
        shutil.rmtree("./faiss_index")
    st.cache_resource.clear()
    st.success("Index deleted and cache cleared — will rebuild on next query.")

# ── Manual FAQ entry ──────────────────────────────────────────────────────────
st.divider()
st.subheader("✏️ Add FAQ Manually")

with st.form("manual_faq"):
    manual_q = st.text_input("Question")
    manual_a = st.text_area("Answer", height=100)
    submitted = st.form_submit_button("Add to Knowledge Base")
    if submitted:
        if manual_q.strip() and manual_a.strip():
            entry = f"\n\nQ: {manual_q.strip()}\nA: {manual_a.strip()}\n"
            with open(PROGRAM_DATA_PATH, "a", encoding="utf-8") as f:
                f.write(entry)
            st.success("Added! Reload the knowledge base to apply.")
        else:
            st.warning("Both question and answer are required.")

# ── Resolved history ──────────────────────────────────────────────────────────
if resolved:
    st.divider()
    with st.expander(f"✅ {len(resolved)} resolved queries (history)"):
        for item in reversed(resolved):
            st.markdown(f"**{item['timestamp']}** — {item['query']}")
            st.caption(f"Answer added: {item.get('answer_added','')[:120]}")
            st.divider()
