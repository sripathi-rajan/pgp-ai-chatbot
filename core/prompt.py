def format_context(chunks) -> str:
    """
    Format a list of retrieved chunks into a single context string for the prompt.

    Chunks may be:
      - LangChain Document objects with metadata (PDF or scraped)
      - Plain strings (legacy path — passed through unchanged)

    PDF chunks (metadata["type"] == "pdf") get a labelled header so the LLM
    knows the provenance of each passage:
        [Source: pgp_brochure_2024.pdf | Category: PGP | Page: 3]
        <chunk text>

    Non-PDF Document objects are included as plain text.
    Each chunk is separated by a blank line.
    """
    parts = []
    for chunk in chunks:
        if hasattr(chunk, "metadata") and chunk.metadata.get("type") == "pdf":
            # Build a rich header for PDF-sourced chunks
            source   = chunk.metadata.get("source", "unknown")
            category = chunk.metadata.get("category", "general").upper()
            page     = chunk.metadata.get("page", "")
            page_tag = f" | Page: {page}" if page else ""
            header   = f"[Source: {source} | Category: {category}{page_tag}]"
            parts.append(f"{header}\n{chunk.page_content}")
        elif hasattr(chunk, "page_content"):
            # Scraped / plain Document — no special header needed
            parts.append(chunk.page_content)
        else:
            # Raw string fallback
            parts.append(str(chunk))
    return "\n\n".join(parts)


def build_prompt(query: str, context: str, chat_history: list, intent: str = "") -> str:
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(f"{role}: {msg}" for role, msg in recent)

    return f"""You are a helpful admissions assistant for Masters' Union programs.

STRICT RULES:
1. Answer using the context provided.
2. If the exact answer isn't stated but can be REASONABLY INFERRED from context, infer it and answer.
   Example: Context says "requires STEM background" → you can infer "non-tech background may face challenges"
   Example: Context says "for working professionals with 3+ years" → you can infer "managers are eligible"
3. Only say "I don't have that information" if the topic is completely absent from context AND cannot be inferred.
4. Never say "I don't have information" for questions answerable by reasoning from eligibility criteria, curriculum, or career outcomes in context.
5. If genuinely unknown, say ONLY: "I don't have that specific detail — please contact pgadmissions@mastersunion.org"
- If the query asks to compare or recommend: compare ALL relevant programs from context, mention fees, duration, and suitability.
- If the query asks for a list: provide the complete list, do not truncate.
- Never invent facts, numbers, dates, or names.
- Format fees with ₹ symbol. Always use markdown bullet points (`- item`) for lists and fee breakdowns.
- For fee tables, use nested bullets: top-level category, then `  - sub-item: ₹amount` indented with 2 spaces.
- Use `**bold**` for section headers and total amounts.
- Never dump fees as one long paragraph — always break them into a bulleted list.
- Adapt your answer length and format to what the question actually needs.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUERY: {query} /no_think

ANSWER:"""
