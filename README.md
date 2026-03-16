# Masters' Union AI Chatbot

A chatbot that answers questions about Masters' Union programmes — fees, admissions, placements, curriculum, and campus life.

## What it does

- Understands questions about **28 programmes** (UG, PGP, Executive, Family Business, Immersion)
- Answers from real programme data using AI (Groq LLaMA)
- Handles follow-up questions with conversation history
- Rejects off-topic queries (cricket scores, weather, jokes, etc.)

## Setup

### Step 1 — Install packages

```bash
pip install -r requirements.txt
```

### Step 2 — Add your Groq API key

On **Mac/Linux**:
```bash
export GROQ_API_KEY=gsk_your_key_here
```

On **Windows**:
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

> Get a free Groq key at console.groq.com

### Step 3 — Index the data

```bash
python scripts/ingest_pdfs.py
```

### Step 4 — Run the chatbot

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Programmes Covered

**Undergraduate**
- Technology & Business Management
- Psychology & Marketing
- Data Science & AI
- Finance & Economics
- Design (MUDS)
- TBM / Psychology & Marketing / DS&AI (Illinois Tech, USA)
- TBM (Griffith University, Australia)

**Postgraduate (PGP)**
- Technology & Business Management
- Technology & Business Management (Young Leaders Cohort)
- Human Resources & Organisation Strategy
- Sports Management & Gaming
- Applied AI & Agentic Systems
- UI/UX & AI Product Design
- Sustainability & Business Management

**Executive**
- PGP Rise: General Management
- Capital Markets and Trading
- Entrepreneurship and Business Acceleration
- PGP Rise: General Management (Global)
- Bloomberg Equity Research Programme

**Family Business**
- PGP Rise: Owners and Promoters Management
- Entrepreneurship and Business Acceleration

**Immersion**
- PGP Bharat
- Bharat Fellowship

---

## Project Files

```
pgp-ai-chatbot/
├── app.py              # Main server (run this)
├── index.html          # Chatbot UI
├── core/
│   ├── pipeline.py     # Loads data and sets up AI
│   ├── retriever.py    # Finds relevant information
│   ├── intent.py       # Understands what the user is asking
│   └── prompt.py       # Builds the AI prompt
├── data/
│   ├── program_data.txt  # Programme details
│   ├── brochure.pdf      # PDF brochure (optional)
│   └── raw/              # Scraped pages
└── scripts/
    └── ingest_pdfs.py    # Indexes PDFs into the search database
```

---

## API Key Safety

- Your Groq API key is **never sent to the browser** — it stays on the server only.
- Other users **cannot read your key**, but if your server is publicly accessible they can send requests that use your quota.
- For local use on your own machine, you are completely safe.

---

## Version

**v1.0.0** — Flask server, hybrid RAG retrieval, Groq LLaMA 3.1 8B, 28 programmes, ~1452 indexed chunks.
