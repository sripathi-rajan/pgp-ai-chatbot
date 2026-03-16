def format_context(chunks) -> str:
    """
    Format retrieved chunks into a context string for the prompt.

    All Document objects (PDF, scraped, or plain) get a rich header:
        [Source: filename | Category: PGP | Type: brochure | Page: 3]

    Raw strings are passed through unchanged.
    Each chunk is separated by a blank line.
    """
    parts = []
    for chunk in chunks:
        if hasattr(chunk, "metadata"):
            meta         = chunk.metadata
            source       = meta.get("source", "unknown")
            category     = meta.get("category", "general").upper()
            content_type = meta.get("content_type", "")
            page         = meta.get("page", "")
            type_tag     = f" | Type: {content_type}" if content_type else ""
            page_tag     = f" | Page: {page}" if page else ""
            header       = f"[Source: {source} | Category: {category}{type_tag}{page_tag}]"
            parts.append(f"{header}\n{chunk.page_content}")
        else:
            # Raw string fallback
            parts.append(str(chunk))
    return "\n\n".join(parts)


def build_prompt(query: str, context: str, chat_history: list, intent: str = "") -> str:
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(f"{role}: {msg}" for role, msg in recent)

    return f"""You are an expert AI assistant for Masters' Union, an industry-focused educational institution in India.

You have access to detailed information about:
- All undergraduate, postgraduate and executive programmes offered at Masters' Union
- Fees, scholarships and financial aid for each programme
- Admission process and eligibility criteria
- Placement statistics and recruiting companies
- Campus life, clubs and student activities
- Faculty and mentors
- Global immersion programmes

RULES:
1. Answer ONLY from the context provided below — do not invent facts, numbers, dates, or names.
2. If listing courses or programmes, list ALL that appear in the context — do not truncate or summarise the list.
3. If the context contains a numbered or bulleted list, preserve that exact format in your answer.
4. If the answer is not in the context, say exactly: "I don't have that detail. Please contact admissions@mastersunion.org or visit mastersunion.org"
5. Never say "I don't have information" for questions clearly answerable by reasoning from eligibility criteria, curriculum, or career outcomes in context.
6. Format fees with ₹ symbol. Use markdown bullet points for lists and fee breakdowns.
7. Use `**bold**` for section headers and important totals.
8. Keep answers concise but complete — do not truncate lists.
9. For course or programme listings, always show the full programme name exactly as it appears in the context.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUERY: {query} /no_think

ANSWER:"""
