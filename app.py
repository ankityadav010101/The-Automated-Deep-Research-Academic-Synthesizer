from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FILES = DATA / "papers"
DB_PATH = DATA / "library.json"
FILES.mkdir(parents=True, exist_ok=True)
env_file = BASE / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

app = FastAPI(title="LitReview — Research Synthesizer", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
MODEL_LAST_ERROR: str | None = None


def read_db() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"documents": [], "reviews": []}
    try:
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"documents": [], "reviews": []}


def write_db(db: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DB_PATH)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[\w-]{3,}", text.lower()) if w not in {
        "the", "and", "for", "with", "that", "this", "from", "were", "was", "are",
        "their", "have", "has", "into", "than", "then", "these", "those", "using",
        "used", "which", "when", "where", "what", "how", "why", "can", "may", "not",
    }}


def parse_pdf(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = fitz.open(path)
    pages, chunks = [], []
    for page_no, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", sort=True)
        text = "\n".join(str(b[4]).strip() for b in blocks if len(b) > 4 and str(b[4]).strip())
        pages.append(text)
        # Keep paragraphs together, then split unusually long blocks without losing page provenance.
        for block_no, block in enumerate(blocks):
            if len(block) <= 4:
                continue
            content = re.sub(r"\s+", " ", str(block[4])).strip()
            if len(content) < 35:
                continue
            pieces = [content[i:i + 1500] for i in range(0, len(content), 1500)]
            for piece_no, piece in enumerate(pieces):
                chunks.append({"id": f"p{page_no}-b{block_no}-{piece_no}", "page": page_no,
                               "text": piece, "bbox": [round(float(x), 1) for x in block[:4]]})
    metadata = doc.metadata or {}
    full = "\n".join(pages)
    title = (metadata.get("title") or "").strip()
    if not title or title.lower() in {"untitled", "unknown"}:
        title = next((line.strip() for line in full.splitlines() if len(line.strip()) > 8), path.stem)
    abstract = ""
    match = re.search(r"abstract\s*[:\n]\s*(.{80,1800}?)(?:\n\s*(?:keywords|introduction|1\.?\s+introduction)\b|$)", full, re.I | re.S)
    if match:
        abstract = re.sub(r"\s+", " ", match.group(1)).strip()
    headings = []
    for line in full.splitlines():
        line = line.strip()
        if 4 < len(line) < 100 and (re.match(r"^(\d+(\.\d+)*\s+)?[A-Z][A-Za-z ,:&()-]+$", line) or line.isupper()):
            if line not in headings:
                headings.append(line)
    return chunks, {"title": title[:240], "author": metadata.get("author", ""),
                    "year": (re.search(r"\b(?:19|20)\d{2}\b", full[:5000]) or [""])[0],
                    "abstract": abstract[:1800], "sections": headings[:30],
                    "page_count": len(doc), "chunks": chunks}


def get_llm():
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if provider == "ollama":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("MODEL_API_KEY", "ollama"),
                      base_url=os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1"))
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=os.getenv("MODEL_BASE_URL") or None)
    except Exception:
        return None


def model_json(system: str, user: str) -> Any | None:
    global MODEL_LAST_ERROR
    client = get_llm()
    if not client:
        MODEL_LAST_ERROR = "No API key was loaded; using local extractive mode."
        return None
    try:
        default_model = "qwen3:4b-instruct" if os.getenv("MODEL_PROVIDER", "openai").lower() == "ollama" else "gpt-4.1-mini"
        result = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", default_model), temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        MODEL_LAST_ERROR = None
        return json.loads(result.choices[0].message.content or "{}")
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        kind = type(exc).__name__
        if status == 429 or kind == "RateLimitError":
            MODEL_LAST_ERROR = "OpenAI returned HTTP 429 (rate limit or project quota). Check the API project’s billing and limits."
        elif status == 401 or kind == "AuthenticationError":
            MODEL_LAST_ERROR = "OpenAI rejected the API key (HTTP 401). Check the key in .env."
        elif status == 403 or kind == "PermissionDeniedError":
            MODEL_LAST_ERROR = "The API key is not allowed to use the configured model (HTTP 403)."
        elif status == 404 or kind == "NotFoundError":
            MODEL_LAST_ERROR = "The configured model was not found (HTTP 404). Check OPENAI_MODEL."
        elif kind == "APIConnectionError":
            service = "the local Ollama server" if os.getenv("MODEL_PROVIDER", "openai").lower() == "ollama" else "the configured model API"
            MODEL_LAST_ERROR = f"The app could not connect to {service}."
        elif status == 400 and os.getenv("MODEL_PROVIDER", "openai").lower() == "ollama":
            MODEL_LAST_ERROR = "Ollama rejected the request. Its context window may be too small; the local review now limits evidence to six passages."
        else:
            MODEL_LAST_ERROR = f"The model request failed ({kind}{f', HTTP {status}' if status else ''})."
        return None


def citations_are_grounded(review: dict[str, Any], evidence_count: int) -> bool:
    """Require valid citations and attach paragraph-end citations to their sentences."""
    valid = set(range(1, evidence_count + 1))
    for section in review.get("sections", []):
        paragraphs = re.split(r"\n\s*\n", normalize_citation_style(str(section.get("body", ""))).strip())
        normalized_paragraphs = []
        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+|\n+", paragraph.strip())
            all_refs = [int(n) for n in re.findall(r"\bS(\d+)\b", paragraph, re.I)]
            if any(n not in valid for n in all_refs):
                return False
            # Small local models often put a paragraph's references at its end.
            # Carry those valid refs to any sentence in that paragraph that lacks them.
            for i, sentence in enumerate(sentences):
                refs = [int(n) for n in re.findall(r"\bS(\d+)\b", sentence, re.I)]
                if not sentence.strip():
                    continue
                if not refs:
                    if not all_refs:
                        return False
                    suffix = " ".join(f"[S{n}]" for n in dict.fromkeys(all_refs))
                    sentences[i] = f"{sentence.rstrip()} {suffix}"
            normalized_paragraphs.append(" ".join(s.strip() for s in sentences if s.strip()))
        section["body"] = "\n\n".join(normalized_paragraphs)
    for item in review.get("disagreements", []):
        refs = item.get("refs", [])
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs:
            numbers = [int(n) for n in re.findall(r"\bS(\d+)\b", str(ref), re.I)]
            if not numbers or any(n not in valid for n in numbers):
                return False
    return True


def normalize_citation_style(text: str) -> str:
    def group(match: re.Match[str]) -> str:
        ids = re.findall(r"\bS(\d+)\b", match.group(0), re.I)
        return " ".join(f"[S{n}]" for n in dict.fromkeys(ids))
    text = re.sub(r"\[(?=[^\]]*\bS\d+\b)[^\]]+\]", group, text, flags=re.I)
    return re.sub(r"(?<![\w\[])S(\d+)(?!\w)", r"[S\1]", text, flags=re.I)


def normalize_subqueries(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, dict):
            candidate = next((item.get(k) for k in ("query", "question", "subquestion", "sub_query", "text", "description")
                              if isinstance(item.get(k), str)), "")
        else:
            candidate = ""
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate and candidate not in result:
            result.append(candidate)
    return result


class ReviewRequest(BaseModel):
    question: str = Field(min_length=10, max_length=2000)
    document_ids: list[str] = Field(default_factory=list)
    max_sources: int = Field(default=24, ge=5, le=60)


class CompareRequest(BaseModel):
    left_document_id: str
    right_document_id: str
    focus: str = Field(default="", max_length=500)
    criteria: list[str] = Field(default_factory=lambda: ["Methods and study design", "Main findings", "Limitations"])


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/status")
def status():
    db = read_db()
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    configured = provider == "ollama" or bool(os.getenv("OPENAI_API_KEY"))
    return {"ready": True, "model_configured": configured,
            "model_provider": provider,
            "model_name": os.getenv("OPENAI_MODEL", "gpt-4.1-mini" if provider == "openai" else "qwen3:4b-instruct"),
            "model_last_error": MODEL_LAST_ERROR,
            "documents": len(db["documents"]), "reviews": len(db["reviews"])}


@app.get("/api/documents")
def documents():
    return [{k: v for k, v in d.items() if k != "chunks"} for d in read_db()["documents"]]


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    db = read_db()
    result = []
    for upload_file in files:
        if not upload_file.filename or not upload_file.filename.lower().endswith(".pdf"):
            continue
        content = await upload_file.read()
        if not content.startswith(b"%PDF"):
            continue
        doc_id = uuid.uuid4().hex[:12]
        path = FILES / f"{doc_id}.pdf"
        path.write_bytes(content)
        try:
            chunks, meta = parse_pdf(path)
            record = {"id": doc_id, "filename": Path(upload_file.filename).name,
                      "size": len(content), "status": "ready" if chunks else "no_text",
                      **meta}
            db["documents"].append(record)
            result.append({k: v for k, v in record.items() if k != "chunks"})
        except Exception as exc:
            path.unlink(missing_ok=True)
            result.append({"filename": upload_file.filename, "status": "error", "error": str(exc)[:180]})
    write_db(db)
    return {"documents": result}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    db = read_db()
    found = next((d for d in db["documents"] if d["id"] == doc_id), None)
    if not found:
        raise HTTPException(404, "Document not found")
    db["documents"] = [d for d in db["documents"] if d["id"] != doc_id]
    write_db(db)
    (FILES / f"{doc_id}.pdf").unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/file")
def pdf_file(doc_id: str):
    if not any(d["id"] == doc_id for d in read_db()["documents"]):
        raise HTTPException(404, "Document not found")
    return FileResponse(FILES / f"{doc_id}.pdf", media_type="application/pdf")


def retrieve(docs: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    qwords = words(query)
    scored = []
    for d in docs:
        for c in d.get("chunks", []):
            cwords = words(c["text"])
            overlap = len(qwords & cwords)
            if not overlap:
                continue
            # BM25-lite ranking with modest title and abstract boosts.
            score = overlap / (0.5 + len(qwords) ** 0.45) + 0.18 * len(qwords & words(d["title"]))
            score += 0.08 * len(qwords & words(d.get("abstract", "")))
            scored.append({**c, "document_id": d["id"], "title": d["title"],
                           "filename": d["filename"], "score": round(score, 4)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]


@app.post("/api/reviews")
def create_review(req: ReviewRequest):
    global MODEL_LAST_ERROR
    MODEL_LAST_ERROR = None
    db = read_db()
    selected = [d for d in db["documents"] if not req.document_ids or d["id"] in req.document_ids]
    if not selected:
        raise HTTPException(400, "Upload at least one paper first")
    if not any(d.get("chunks") for d in selected):
        raise HTTPException(400, "No searchable text found. Scanned papers need OCR before review.")
    expansion = model_json(
        "Expand the research question into 4-6 distinct, searchable subquestions. Return JSON {subqueries:[...]}. Keep the original scope; no external facts.",
        req.question)
    subqueries = normalize_subqueries(expansion.get("subqueries", [])) if expansion else []
    if not subqueries:
        subqueries = [req.question, f"methods and measurements: {req.question}",
                      f"limitations and conflicting findings: {req.question}",
                      f"future research and open questions: {req.question}"]
    evidence, seen = [], set()
    # Qwen3 4B is commonly run with a 4K context window. Six passages keep
    # the draft prompt within that window for the PDFs indexed by this app.
    source_limit = min(req.max_sources, 6) if os.getenv("MODEL_PROVIDER", "openai").lower() == "ollama" else req.max_sources
    per_query = max(2, source_limit // max(1, len(subqueries)))
    for sq in subqueries:
        for hit in retrieve(selected, sq, per_query):
            if hit["id"] not in seen:
                seen.add(hit["id"])
                hit["subquery"] = sq
                evidence.append(hit)
    evidence.sort(key=lambda x: x["score"], reverse=True)
    evidence = evidence[:source_limit]
    if not evidence:
        raise HTTPException(400, "No passages matched. Try a more specific question or upload papers with extractable text.")

    compact = [{"ref": f"[S{i+1}]", "paper": e["title"], "filename": e["filename"],
                "page": e["page"], "text": e["text"], "bbox": e["bbox"]}
               for i, e in enumerate(evidence)]
    draft_data = model_json(
        "You are a careful academic literature-review assistant. Use only supplied evidence. Write at most 3 concise sections, each with 2-4 sentences. Every factual sentence must end with one or more exact citations in the form [S1] (use only supplied S numbers). Never infer study details. Distinguish agreement from disagreement and note limits. Keep each sentence short. Return JSON: {title, summary, sections:[{heading,body}], disagreements:[{claim,refs:[S1],note}], gaps:[...], limitations:[...]}",
        json.dumps({"question": req.question, "subquestions": subqueries, "evidence": compact}, ensure_ascii=False))
    if draft_data and draft_data.get("sections") and citations_are_grounded(draft_data, len(evidence)):
        review = draft_data
        mode = "model"
    else:
        if draft_data and draft_data.get("sections"):
            MODEL_LAST_ERROR = "The model draft did not pass the inline citation check, so source-only text was used instead."
        sections = []
        for sq in subqueries[:4]:
            hits = [e for e in evidence if e["subquery"] == sq][:4]
            if not hits:
                continue
            lines = [f"{e['text'].strip()} [S{evidence.index(e)+1}]" for e in hits]
            sections.append({"heading": sq, "body": "\n\n".join(lines)})
        review = {"title": "Literature review: " + req.question[:90],
                  "summary": "Evidence-led passage synthesis. Configure OPENAI_API_KEY for cohesive model-assisted synthesis; all passages below remain linked to their source pages.",
                  "sections": sections, "disagreements": [],
                  "gaps": ["Cross-study agreement and contradiction analysis needs model-assisted review."],
                  "limitations": ["The local fallback retrieves by lexical overlap and quotes source passages; it does not independently infer study findings."]}
        mode = "extractive"
    if mode == "extractive":
        review["warning"] = MODEL_LAST_ERROR or "Model synthesis was unavailable; this review uses retrieved source passages."

    # Compare documents with opposing language on similarly retrieved evidence; always label as a review flag.
    if not review.get("disagreements"):
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for e in evidence:
            by_doc.setdefault(e["document_id"], []).append(e)
        positive = re.compile(r"\b(improv|increase|benefit|effective|significant|support|improve|higher)\w*\b", re.I)
        negative = re.compile(r"\b(no effect|not significant|decrease|worsen|ineffective|fail|decline|lower)\w*\b", re.I)
        flags = []
        docs_hits = list(by_doc.items())
        for i, (doc_a, hits_a) in enumerate(docs_hits):
            for doc_b, hits_b in docs_hits[i+1:]:
                a = next((h for h in hits_a if positive.search(h["text"])), None)
                b = next((h for h in hits_b if negative.search(h["text"])), None)
                if a and b:
                    flags.append({"claim": "Potentially contrasting outcome language detected", "refs": [f"S{evidence.index(a)+1}", f"S{evidence.index(b)+1}"],
                                  "note": "Automated lexical flag only; inspect study populations, outcomes, and methods before treating these papers as contradictory."})
                    if len(flags) >= 5:
                        break
        review["disagreements"] = flags
    review["mode"] = mode
    review["subqueries"] = subqueries
    review["evidence"] = [{**c, "document_id": evidence[i]["document_id"],
                           "chunk_id": evidence[i]["id"], "score": evidence[i]["score"],
                           "subquery": evidence[i]["subquery"]} for i, c in enumerate(compact)]
    review["created_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    review["id"] = uuid.uuid4().hex[:12]
    db["reviews"].append(review)
    write_db(db)
    return review


@app.post("/api/compare")
def compare_papers(req: CompareRequest):
    global MODEL_LAST_ERROR
    MODEL_LAST_ERROR = None
    if req.left_document_id == req.right_document_id:
        raise HTTPException(400, "Choose two different papers to compare.")
    db = read_db()
    by_id = {d["id"]: d for d in db["documents"]}
    left_doc = by_id.get(req.left_document_id)
    right_doc = by_id.get(req.right_document_id)
    if not left_doc or not right_doc:
        raise HTTPException(404, "One of the selected papers is no longer in the library.")
    if not left_doc.get("chunks") or not right_doc.get("chunks"):
        raise HTTPException(400, "Both papers need searchable text. Scanned PDFs need OCR before comparison.")

    criteria = []
    for item in req.criteria[:3]:
        value = re.sub(r"\s+", " ", str(item)).strip()[:100]
        if value and value not in criteria:
            criteria.append(value)
    if not criteria:
        criteria = ["Methods and study design", "Main findings", "Limitations"]

    query_terms = {
        "method": "method design approach participants sample procedure",
        "design": "method design approach participants sample procedure",
        "finding": "results findings outcome effect measurement",
        "result": "results findings outcome effect measurement",
        "limitation": "limitations weakness constraint future work",
        "future": "limitations weakness constraint future work",
    }
    evidence: list[dict[str, Any]] = []
    row_evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for criterion in criteria:
        terms = " ".join(v for key, v in query_terms.items() if key in criterion.lower())
        query = " ".join(part for part in (req.focus.strip(), criterion, terms) if part)
        sides = []
        for doc in (left_doc, right_doc):
            hits = retrieve([doc], query, 2)
            hit = next((h for h in hits if f"{doc['id']}:{h['id']}" not in seen), hits[0] if hits else None)
            if hit:
                seen.add(f"{doc['id']}:{hit['id']}")
                hit["subquery"] = criterion
                evidence.append(hit)
                sides.append(f"[S{len(evidence)}]")
            else:
                sides.append(None)
        row_evidence.append({"criterion": criterion, "left_ref": sides[0], "right_ref": sides[1]})

    compact = [{"ref": f"[S{i+1}]", "paper": e["title"], "page": e["page"], "text": e["text"]}
               for i, e in enumerate(evidence)]
    comparison = model_json(
        "Compare exactly the two named papers in the supplied evidence. Do not change the topic or introduce tools or facts not in the papers. Return JSON with comparison_rows, one row for each supplied criterion, each shaped {criterion,left,right,difference}. left must use evidence only from the first paper; right only from the second; difference may use both. Do not write citation labels; the app attaches source citations to each column. If the supplied passages do not establish a detail, say 'Not reported in the retrieved passages.' Do not guess. Keep cells concise.",
        json.dumps({"focus": req.focus, "paper_left": left_doc["title"], "paper_right": right_doc["title"],
                    "criteria": row_evidence, "evidence": compact}, ensure_ascii=False)) if evidence else None

    model_rows = comparison.get("comparison_rows", []) if comparison else []
    model_ok = isinstance(model_rows, list) and len(model_rows) == len(criteria) and all(isinstance(row, dict) for row in model_rows)

    rows = []
    if model_ok:
        for i, row in enumerate(model_rows):
            expected = row_evidence[i]
            def attach_source(text: Any, refs: list[str]) -> str:
                clean = re.sub(r"\[?S\d+\]?", "", str(text), flags=re.I)
                clean = re.sub(r"\s+", " ", clean).strip()
                return (clean + (" " if clean else "") + " ".join(refs)).strip()
            left_cell = attach_source(row.get("left", ""), [expected["left_ref"]]) if expected["left_ref"] else "No matching passage was found in the retrieved text."
            right_cell = attach_source(row.get("right", ""), [expected["right_ref"]]) if expected["right_ref"] else "No matching passage was found in the retrieved text."
            refs = [ref for ref in (expected["left_ref"], expected["right_ref"]) if ref]
            difference = attach_source(row.get("difference", ""), refs) if len(refs) == 2 else "Insufficient retrieved evidence from both papers to compare this dimension."
            rows.append({"criterion": str(row.get("criterion", criteria[i])),
                         "left": left_cell, "right": right_cell, "difference": difference})
        mode = "model"
    else:
        if comparison:
            MODEL_LAST_ERROR = "The model comparison did not return valid citations for both papers; showing the retrieved passages instead."
        for item in row_evidence:
            def quote(ref: str | None) -> str:
                if not ref:
                    return "No matching passage was found in the retrieved text."
                ev = evidence[int(ref[2:-1]) - 1]
                return f"{ev['text'].strip()} {ref}"
            rows.append({"criterion": item["criterion"], "left": quote(item["left_ref"]),
                         "right": quote(item["right_ref"]),
                         "difference": "Compare the cited passages directly; model synthesis was unavailable."})
        mode = "extractive"

    review = {
        "id": uuid.uuid4().hex[:12],
        "title": f"Paired comparison: {left_doc['title']} vs. {right_doc['title']}",
        "summary": f"Side-by-side comparison of {len(criteria)} criteria. Each cell is tied to passages retrieved from its paper.",
        "comparison_rows": rows,
        "comparison_papers": [{"id": left_doc["id"], "title": left_doc["title"]},
                              {"id": right_doc["id"], "title": right_doc["title"]}],
        "evidence": [{**compact[i], "document_id": evidence[i]["document_id"], "chunk_id": evidence[i]["id"],
                      "score": evidence[i]["score"], "subquery": evidence[i]["subquery"]}
                     for i in range(len(evidence))],
        "mode": mode,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    if mode == "extractive":
        review["warning"] = MODEL_LAST_ERROR or "No model output was available; retrieved passages are shown verbatim."
    db["reviews"].append(review)
    write_db(db)
    return review


@app.get("/api/reviews")
def reviews():
    return [{"id": r["id"], "title": r["title"], "created_at": r["created_at"], "mode": r["mode"]}
            for r in reversed(read_db()["reviews"])]


@app.get("/api/reviews/{review_id}")
def get_review(review_id: str):
    review = next((r for r in read_db()["reviews"] if r["id"] == review_id), None)
    if not review:
        raise HTTPException(404, "Review not found")
    return review


@app.get("/api/reviews/{review_id}/markdown")
def markdown(review_id: str):
    review = next((r for r in read_db()["reviews"] if r["id"] == review_id), None)
    if not review:
        raise HTTPException(404, "Review not found")
    out = [f"# {review['title']}", "", review.get("summary", ""), ""]
    if review.get("comparison_rows"):
        papers = review.get("comparison_papers", [{"title": "Paper A"}, {"title": "Paper B"}])
        out.extend([f"| Criterion | {papers[0]['title']} | {papers[1]['title']} | Difference |",
                    "|---|---|---|---|"])
        for row in review["comparison_rows"]:
            cells = [row.get(k, "").replace("|", "\\|").replace("\n", " ") for k in ("criterion", "left", "right", "difference")]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    for section in review.get("sections", []):
        out.extend([f"## {section['heading']}", "", section["body"], ""])
    for label, key in [("Potential disagreements to inspect", "disagreements"), ("Research gaps", "gaps"), ("Limitations", "limitations")]:
        items = review.get(key, [])
        if items:
            out.extend([f"## {label}", ""])
            for item in items:
                out.append("- " + (item.get("note", item.get("claim", "")) if isinstance(item, dict) else item))
            out.append("")
    return {"markdown": "\n".join(out)}
