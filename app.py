from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pymupdf as fitz
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FILES = DATA / "papers"
STICKY_IMAGES = DATA / "sticky-images"
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
        return {"documents": [], "reviews": [], "chats": [], "annotations": [], "sticky_notes": []}
    try:
        db = json.loads(DB_PATH.read_text(encoding="utf-8"))
        for key in ("documents", "reviews", "chats", "annotations", "sticky_notes"):
            db.setdefault(key, [])
        return db
    except (json.JSONDecodeError, OSError):
        return {"documents": [], "reviews": [], "chats": [], "annotations": [], "sticky_notes": []}


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
            max_tokens=1800,
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
    document_ids: list[str] = Field(min_length=2, max_length=8)
    focus: str = Field(default="", max_length=500)
    criteria: list[str] = Field(default_factory=lambda: ["Methods and study design", "Main findings", "Limitations"])


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)


class MultiPaperRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] = Field(min_length=2, max_length=8)


class ClaimVerificationRequest(BaseModel):
    claim_index: int = Field(ge=0, le=4)
    paper_index: int = Field(ge=0, le=7)
    status: Literal["supported", "contradicted", "not_addressed", "unclear"]
    note: str = Field(default="", max_length=1000)
    study_context: str = Field(default="", max_length=600)


class CoverageRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] = Field(min_length=1, max_length=20)


class SavedChatRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    messages: list[dict[str, Any]] = Field(default_factory=list, max_length=100)


class AnnotationRequest(BaseModel):
    document_id: str
    page: int = Field(ge=1)
    kind: Literal["note", "bookmark", "highlight"] = "note"
    body: str = Field(default="", max_length=2000)
    selected_text: str = Field(default="", max_length=2000)
    chunk_id: str = Field(default="", max_length=100)


class StickyNoteUpdate(BaseModel):
    body: str = Field(default="", max_length=4000)
    color: Literal["yellow", "pink", "blue", "green"] = "yellow"


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


@app.get("/api/documents/{doc_id}/pages/{page_no}/image")
def pdf_page_image(doc_id: str, page_no: int):
    document = next((d for d in read_db()["documents"] if d["id"] == doc_id), None)
    if not document:
        raise HTTPException(404, "Document not found")
    if page_no < 1 or page_no > document.get("page_count", 0):
        raise HTTPException(400, "Page number is outside this PDF")
    path = FILES / f"{doc_id}.pdf"
    with fitz.open(path) as pdf:
        page = pdf.load_page(page_no - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        return Response(content=pixmap.tobytes("png"), media_type="image/png",
                        headers={"Cache-Control": "no-store"})


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
    if len(set(req.document_ids)) != len(req.document_ids):
        raise HTTPException(400, "Choose each paper only once.")
    db = read_db()
    by_id = {d["id"]: d for d in db["documents"]}
    selected = [by_id.get(doc_id) for doc_id in req.document_ids]
    if any(doc is None for doc in selected):
        raise HTTPException(404, "One of the selected papers is no longer in the library.")
    papers = [doc for doc in selected if doc is not None]
    if any(not doc.get("chunks") for doc in papers):
        raise HTTPException(400, "Every selected paper needs searchable text. Scanned PDFs need OCR before comparison.")

    criteria = []
    for item in req.criteria[:4]:
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
    for criterion in criteria:
        terms = " ".join(v for key, v in query_terms.items() if key in criterion.lower())
        query = " ".join(part for part in (req.focus.strip(), criterion, terms) if part)
        cells = []
        for doc in papers:
            hits = retrieve([doc], query, 1)
            hit = hits[0] if hits else None
            if hit:
                hit["subquery"] = criterion
                evidence.append(hit)
                cells.append(f"[S{len(evidence)}]")
            else:
                cells.append(None)
        row_evidence.append({"criterion": criterion, "refs": cells})

    ollama = os.getenv("MODEL_PROVIDER", "openai").lower() == "ollama"
    compact = [{"ref": f"[S{i+1}]", "document_id": e["document_id"],
                "paper": e["title"], "page": e["page"], "text": e["text"][:700]}
               for i, e in enumerate(evidence)]
    model_compact = [{**item, "text": item["text"][:240] if ollama else item["text"]}
                     for item in compact]
    comparison = model_json(
        "Compare the named academic papers using only the supplied passages. Return JSON with comparison_rows, one row per supplied criterion, each shaped {criterion,cells:[...]}. Each row's cells must have one concise string per paper, in the exact paper order supplied. Never transfer a fact from one paper to another. Do not include citation labels; the app attaches citations to each cell. If evidence does not establish a detail, say 'Not reported in the retrieved passages.' Do not guess.",
        json.dumps({"focus": req.focus, "papers": [{"id": d["id"], "title": d["title"]} for d in papers],
                    "criteria": row_evidence, "evidence": model_compact}, ensure_ascii=False)) if evidence else None

    model_rows = comparison.get("comparison_rows", []) if isinstance(comparison, dict) else []
    model_ok = (isinstance(model_rows, list) and len(model_rows) == len(criteria) and
                all(isinstance(row, dict) and isinstance(row.get("cells"), list) and
                    len(row["cells"]) == len(papers) for row in model_rows))

    rows = []
    if model_ok:
        for i, row in enumerate(model_rows):
            expected = row_evidence[i]
            cells = []
            for j, value in enumerate(row["cells"]):
                ref = expected["refs"][j]
                clean = re.sub(r"\[?S\d+\]?", "", str(value))
                clean = re.sub(r"\s+", " ", clean).strip()
                cells.append((clean + (" " if clean else "") + ref).strip() if ref else
                             "No matching passage was found in the retrieved text.")
            rows.append({"criterion": criteria[i], "cells": cells})
        mode = "model"
    else:
        if comparison:
            MODEL_LAST_ERROR = "The model comparison did not return a valid cell for every paper; showing the retrieved passages instead."
        for item in row_evidence:
            def quote(ref: str | None) -> str:
                if not ref:
                    return "No matching passage was found in the retrieved text."
                ev = evidence[int(ref[2:-1]) - 1]
                return f"{ev['text'].strip()} {ref}"
            rows.append({"criterion": item["criterion"], "cells": [quote(ref) for ref in item["refs"]]})
        mode = "extractive"

    summary = (f"Evidence-led comparison of {len(papers)} papers across {len(criteria)} dimensions. "
               "Each paper column is grounded in retrieved passages from that source.")
    review = {
        "id": uuid.uuid4().hex[:12],
        "title": f"Comparison across {len(papers)} papers",
        "summary": summary,
        "comparison_rows": rows,
        "comparison_papers": [{"id": d["id"], "title": d["title"]} for d in papers],
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


@app.post("/api/chat")
def chat(req: ChatRequest):
    global MODEL_LAST_ERROR
    MODEL_LAST_ERROR = None
    db = read_db()
    requested_ids = set(req.document_ids)
    selected = [d for d in db["documents"] if d.get("status") == "ready" and d.get("chunks") and
                (not requested_ids or d["id"] in requested_ids)]
    if not selected:
        raise HTTPException(400, "Select at least one indexed paper with searchable text.")

    # Include the latest user turn in lexical retrieval so follow-up questions like
    # “what about the limitations?” retain some context about the topic.
    previous_question = next((turn.content for turn in reversed(req.history) if turn.role == "user"), "")
    search_query = " ".join((previous_question, req.question)).strip()
    ollama = os.getenv("MODEL_PROVIDER", "openai").lower() == "ollama"
    # Give local models enough evidence to cover each selected paper separately.
    source_limit = min(12, max(8, len(selected) * 2)) if ollama else 10
    ranked = retrieve(selected, search_query, max(source_limit, min(len(selected) * 2, 40)))
    evidence, per_doc = [], Counter()
    for hit in ranked:
        if per_doc[hit["document_id"]] >= 2:
            continue
        evidence.append(hit)
        per_doc[hit["document_id"]] += 1
        if len(evidence) >= source_limit:
            break
    if not evidence:
        raise HTTPException(400, "No passages matched. Try a more specific question or select other papers.")

    compact = [{"ref": f"[S{i+1}]", "paper": e["title"], "page": e["page"], "text": e["text"][:900]}
               for i, e in enumerate(evidence)]
    turns = [{"role": t.role, "content": t.content[:800]} for t in req.history[-4:]]
    model_compact = [{**item, "text": item["text"][:1000] if ollama else item["text"]}
                     for item in compact]
    result = model_json(
        "You answer questions about a selected academic paper library. Use only the supplied passages for factual claims. "
        "Cite each factual claim inline using the supplied labels exactly, such as [S1]. Do not invent citations or outside facts. "
        "If the user asks to compare or discuss papers separately, make a clearly titled subsection for EVERY selected paper, "
        "in the supplied paper order. For each paper, explain its relevant approach, findings, and limitations when the evidence supports them; "
        "then add a short cross-paper comparison explaining similarities and differences. Never attribute one paper's evidence to another. "
        "Use several substantive sentences per paper when requested, and do not compress the whole answer into a few generic paragraphs. "
        "If evidence is missing for a detail, say so. If the passages do not answer the question, say so clearly. "
        "Return JSON {answer: string}; use readable headings and paragraphs.",
        json.dumps({"question": req.question, "conversation": turns,
                    "selected_papers": [d["title"] for d in selected], "evidence": model_compact}, ensure_ascii=False))
    answer = str(result.get("answer", "")).strip() if isinstance(result, dict) else ""
    answer = normalize_citation_style(answer)
    valid_refs = {f"S{i+1}" for i in range(len(evidence))}
    refs = set(re.findall(r"\[(S\d+)\]", answer))
    grounded = bool(answer and refs and refs.issubset(valid_refs))
    if not grounded:
        if result and answer:
            MODEL_LAST_ERROR = "The chat answer did not include valid inline citations, so retrieved source passages are shown instead."
        answer = "I couldn't produce a grounded synthesis for this turn. Here are the most relevant passages I found:\n\n" + \
            "\n\n".join(f"{e['title']} (p. {e['page']}): {e['text'].strip()} [S{i+1}]"
                            for i, e in enumerate(evidence))
    return {"answer": answer, "warning": MODEL_LAST_ERROR,
            "evidence": [{**compact[i], "document_id": evidence[i]["document_id"],
                          "chunk_id": evidence[i]["id"]} for i in range(len(evidence))]}


@app.get("/api/chats")
def saved_chats():
    return [{"id": c["id"], "title": c["title"], "updated_at": c["updated_at"]}
            for c in sorted(read_db()["chats"], key=lambda c: c.get("updated_at", ""), reverse=True)]


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    chat_record = next((c for c in read_db()["chats"] if c["id"] == chat_id), None)
    if not chat_record:
        raise HTTPException(404, "Chat not found")
    return chat_record


@app.post("/api/chats")
def create_chat(req: SavedChatRequest):
    db = read_db()
    first_question = next((str(m.get("content", "")) for m in req.messages if m.get("role") == "user"), "Research chat")
    record = {"id": uuid.uuid4().hex[:12], "title": first_question[:90],
              "document_ids": req.document_ids, "messages": req.messages,
              "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    db["chats"].append(record)
    write_db(db)
    return record


@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: str, req: SavedChatRequest):
    db = read_db()
    record = next((c for c in db["chats"] if c["id"] == chat_id), None)
    if not record:
        raise HTTPException(404, "Chat not found")
    first_question = next((str(m.get("content", "")) for m in req.messages if m.get("role") == "user"), record["title"])
    record.update({"title": first_question[:90], "document_ids": req.document_ids,
                   "messages": req.messages,
                   "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
    write_db(db)
    return record


@app.post("/api/claims")
def claim_matrix(req: MultiPaperRequest):
    global MODEL_LAST_ERROR
    MODEL_LAST_ERROR = None
    if len(set(req.document_ids)) != len(req.document_ids):
        raise HTTPException(400, "Choose each paper only once.")
    db = read_db()
    by_id = {d["id"]: d for d in db["documents"]}
    papers = [by_id.get(doc_id) for doc_id in req.document_ids]
    if any(d is None for d in papers):
        raise HTTPException(404, "One of the selected papers is no longer in the library.")
    papers = [d for d in papers if d is not None]
    if any(not d.get("chunks") for d in papers):
        raise HTTPException(400, "Every selected paper needs searchable text.")
    evidence = []
    for paper in papers:
        for hit in retrieve([paper], req.question, 2):
            hit["subquery"] = req.question
            evidence.append(hit)
    if not evidence:
        raise HTTPException(400, "No passages matched the question in the selected papers.")
    compact = [{"ref": f"[S{i+1}]", "document_id": e["document_id"], "paper": e["title"],
                "page": e["page"], "text": e["text"][:300]}
               for i, e in enumerate(evidence)]
    ollama = os.getenv("MODEL_PROVIDER", "openai").lower() == "ollama"
    prompt_evidence = [{**item, "text": item["text"][:220] if ollama else item["text"]}
                       for item in compact]
    result = model_json(
        "Extract up to five specific, evidence-checkable claims relevant to the question. Use only the supplied excerpts. "
        "For every paper, classify each claim as supports, challenges, not_addressed, or unclear. Use challenges only when the paper explicitly reports a conflicting result; do not infer contradiction from different topics or methods. "
        "Return exactly one claim row. Use a concise version of the user's question as the claim. That row must contain exactly one assessment for every supplied paper, in the same order as the papers list, including papers whose excerpts do not address it. Return JSON {claims:[{claim, assessments:[{status,assessment}]}]}. Do not create separate claims per paper and do not include paper IDs. Keep each assessment short and do not invent details.",
        json.dumps({"question": req.question, "papers": [{"id": p["id"], "title": p["title"]} for p in papers],
                    "evidence": prompt_evidence}, ensure_ascii=False))
    raw_claims = result.get("claims", result.get("claim_rows", result.get("rows", []))) if isinstance(result, dict) else []
    if isinstance(raw_claims, dict):
        raw_claims = [raw_claims]
    valid_statuses = {"supports", "challenges", "not_addressed", "unclear"}
    def ordered_assessments(row: dict[str, Any]) -> list[Any]:
        values = row.get("assessments", row.get("cells", row.get("papers", [])))
        if isinstance(values, dict):
            return [values.get(p["id"], values.get(p["title"], {})) for p in papers]
        return values if isinstance(values, list) else []
    standard_shape = isinstance(raw_claims, list) and raw_claims and all(
        isinstance(row, dict) and isinstance(ordered_assessments(row), list) and
        len(ordered_assessments(row)) == len(papers) for row in raw_claims[:5])
    if not standard_shape and isinstance(raw_claims, list):
        # Small local models sometimes emit one paper-specific row per paper.
        # Recover those assessments by matching paper-title words in each claim.
        grouped: list[list[dict[str, Any]]] = [[] for _ in papers]
        for row in raw_claims[:8]:
            if not isinstance(row, dict):
                continue
            values = ordered_assessments(row)
            if len(values) != 1:
                continue
            claim_words = words(str(row.get("claim", row.get("statement", ""))))
            overlaps = [len(claim_words & words(p["title"])) for p in papers]
            best = max(overlaps, default=0)
            if best < 3 or overlaps.count(best) != 1:
                continue
            index = overlaps.index(best)
            assessment = values[0]
            if isinstance(assessment, str):
                assessment = {"assessment": assessment}
            if isinstance(assessment, dict):
                grouped[index].append(assessment)
        if all(grouped):
            collapsed = []
            for entries in grouped:
                statuses = []
                descriptions = []
                for entry in entries:
                    raw_status = str(entry.get("status", "unclear")).strip().lower().replace(" ", "_")
                    aliases = {"supported": "supports", "support": "supports", "contradicts": "challenges",
                               "contradicted": "challenges", "conflicts": "challenges", "not_found": "not_addressed",
                               "no_evidence": "not_addressed", "mixed": "unclear"}
                    raw_status = aliases.get(raw_status, raw_status)
                    statuses.append(raw_status if raw_status in valid_statuses else "unclear")
                    text = str(entry.get("assessment", entry.get("evidence", entry.get("reason", "")))).strip()
                    if text and text not in descriptions:
                        descriptions.append(text)
                status_counts = Counter(statuses)
                status = status_counts.most_common(1)[0][0] if len(status_counts) == 1 else "unclear"
                collapsed.append({"status": status, "assessment": " ".join(descriptions[:2])[:500]})
            raw_claims = [{"claim": req.question, "assessments": collapsed}]
            standard_shape = True
    valid = bool(standard_shape)
    rows = []
    if valid:
        for row in raw_claims[:5]:
            cells = []
            for assessment in ordered_assessments(row):
                if isinstance(assessment, str):
                    assessment = {"assessment": assessment}
                if not isinstance(assessment, dict):
                    assessment = {}
                status = assessment.get("status", "unclear") if isinstance(assessment, dict) else "unclear"
                status_text = str(status).strip().lower().replace(" ", "_")
                status_aliases = {"supported": "supports", "support": "supports", "supports": "supports",
                                  "contradicts": "challenges", "contradicted": "challenges", "conflicts": "challenges",
                                  "not_addressed": "not_addressed", "not_found": "not_addressed", "no_evidence": "not_addressed",
                                  "unclear": "unclear", "mixed": "unclear"}
                status = status_aliases.get(status_text, status_text)
                status = status if status in valid_statuses else "unclear"
                text = str(assessment.get("assessment", ""))[:500] if isinstance(assessment, dict) else ""
                if not text:
                    text = str(assessment.get("evidence", assessment.get("reason", assessment.get("summary", ""))))[:500]
                paper = papers[len(cells)]
                refs = [f"[S{i+1}]" for i, e in enumerate(evidence) if e["document_id"] == paper["id"]]
                cells.append({"status": status, "text": text or "No specific assessment returned.", "refs": refs})
            rows.append({"claim": str(row.get("claim", "Claim"))[:500], "cells": cells})
    else:
        if result:
            MODEL_LAST_ERROR = "The model did not return a valid claim matrix; showing retrieved evidence for manual review."
        cells = []
        for paper in papers:
            matching = [(i, e) for i, e in enumerate(evidence) if e["document_id"] == paper["id"]]
            cells.append({"status": "unclear",
                          "text": "Model assessment unavailable. Inspect the retrieved passage directly." if matching else "No matching passage was retrieved.",
                          "refs": [f"[S{i+1}]" for i, _ in matching]})
        rows = [{"claim": req.question, "cells": cells}]
    review = {"id": uuid.uuid4().hex[:12], "title": "Claim and evidence matrix",
              "summary": f"Evidence map for {len(papers)} papers. Labels are model assessments to verify against the cited passages.",
              "claim_rows": rows, "comparison_papers": [{"id": p["id"], "title": p["title"]} for p in papers],
              "evidence": [{**compact[i], "chunk_id": evidence[i]["id"]} for i in range(len(evidence))],
              "mode": "model" if valid else "extractive",
              "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    if not valid:
        review["warning"] = MODEL_LAST_ERROR or "Model analysis unavailable; inspect retrieved passages manually."
    db["reviews"].append(review)
    write_db(db)
    return review


@app.put("/api/reviews/{review_id}/verification")
def verify_claim(review_id: str, req: ClaimVerificationRequest):
    db = read_db()
    review = next((r for r in db["reviews"] if r["id"] == review_id), None)
    if not review or not review.get("claim_rows"):
        raise HTTPException(404, "Claim review not found")
    if req.claim_index >= len(review["claim_rows"]):
        raise HTTPException(404, "Claim row not found")
    row = review["claim_rows"][req.claim_index]
    cells = row.get("cells", [])
    if req.paper_index >= len(cells):
        raise HTTPException(404, "Paper assessment not found")
    verification = {"status": req.status, "note": req.note.strip(),
                    "study_context": req.study_context.strip(),
                    "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    cells[req.paper_index]["verification"] = verification
    write_db(db)
    return verification


@app.post("/api/coverage")
def question_coverage(req: CoverageRequest):
    db = read_db()
    by_id = {d["id"]: d for d in db["documents"]}
    selected = [by_id.get(doc_id) for doc_id in req.document_ids]
    if any(d is None for d in selected):
        raise HTTPException(404, "One of the selected papers is no longer in the library.")
    papers = [d for d in selected if d is not None]
    if any(not d.get("chunks") for d in papers):
        raise HTTPException(400, "Every selected paper needs searchable text.")
    expansion = model_json(
        "Break this research question into four short, distinct evidence-searchable subquestions. Return JSON {subqueries:[string]}. Do not answer the question.",
        req.question)
    queries = normalize_subqueries(expansion.get("subqueries", []))[:5] if isinstance(expansion, dict) else []
    if not queries:
        queries = [req.question, f"methods and study design: {req.question}",
                   f"main findings and outcomes: {req.question}", f"limitations and future work: {req.question}"]
    evidence, rows = [], []
    for query in queries:
        cells = []
        for paper in papers:
            hits = retrieve([paper], query, 1)
            if hits:
                hit = hits[0]
                evidence.append(hit)
                cells.append({"found": True, "ref": f"[S{len(evidence)}]"})
            else:
                cells.append({"found": False, "ref": None})
        rows.append({"question": query, "cells": cells})
    review = {"id": uuid.uuid4().hex[:12], "title": "Research question coverage map",
              "summary": "Shows where lexical retrieval found a relevant passage for each subquestion and paper. A match indicates possible coverage, not that the question is fully answered.",
              "coverage_rows": rows, "comparison_papers": [{"id": p["id"], "title": p["title"]} for p in papers],
              "evidence": [{"ref": f"[S{i+1}]", "paper": e["title"], "page": e["page"], "text": e["text"],
                            "document_id": e["document_id"], "chunk_id": e["id"]} for i, e in enumerate(evidence)],
              "mode": "retrieval", "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    db["reviews"].append(review)
    write_db(db)
    return review


@app.get("/api/annotations")
def annotations(document_id: str | None = None, page: int | None = None):
    items = read_db()["annotations"]
    if document_id:
        items = [item for item in items if item["document_id"] == document_id]
    if page is not None:
        items = [item for item in items if item["page"] == page]
    return items


@app.post("/api/annotations")
def add_annotation(req: AnnotationRequest):
    db = read_db()
    doc = next((d for d in db["documents"] if d["id"] == req.document_id), None)
    if not doc:
        raise HTTPException(404, "Paper not found")
    if req.page > doc.get("page_count", 0):
        raise HTTPException(400, "Page number is outside this PDF")
    if req.kind != "bookmark" and not req.body.strip() and not req.selected_text.strip():
        raise HTTPException(400, "Add a note or select a passage before saving.")
    record = {"id": uuid.uuid4().hex[:12], "document_id": req.document_id, "page": req.page,
              "kind": req.kind, "body": req.body.strip(), "selected_text": req.selected_text.strip(),
              "chunk_id": req.chunk_id,
              "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    db["annotations"].append(record)
    write_db(db)
    return record


@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: str):
    db = read_db()
    before = len(db["annotations"])
    db["annotations"] = [item for item in db["annotations"] if item["id"] != annotation_id]
    if len(db["annotations"]) == before:
        raise HTTPException(404, "Annotation not found")
    write_db(db)
    return {"deleted": annotation_id}


def public_sticky_note(note: dict[str, Any], db: dict[str, Any]) -> dict[str, Any]:
    document = next((d for d in db["documents"] if d["id"] == note.get("document_id")), None)
    return {**note, "image_url": f"/api/sticky-notes/{note['id']}/image" if note.get("image_filename") else None,
            "document_title": document.get("title") if document else None,
            "document_exists": bool(document)}


@app.get("/api/sticky-notes")
def list_sticky_notes():
    db = read_db()
    notes = sorted(db["sticky_notes"], key=lambda n: n.get("updated_at", n.get("created_at", "")), reverse=True)
    return [public_sticky_note(note, db) for note in notes]


@app.post("/api/sticky-notes")
async def create_sticky_note(
    body: str = Form(...),
    color: str = Form("yellow"),
    document_id: str = Form(""),
    page: int = Form(0),
    screenshot: UploadFile | None = File(None),
):
    color = color if color in {"yellow", "pink", "blue", "green"} else "yellow"
    body = body.strip()
    if not body and not screenshot:
        raise HTTPException(400, "Add note text or attach a screenshot.")
    db = read_db()
    document = None
    if document_id:
        document = next((d for d in db["documents"] if d["id"] == document_id), None)
        if not document:
            raise HTTPException(404, "Selected paper was not found.")
        if page < 1 or page > document.get("page_count", 0):
            raise HTTPException(400, "Choose a valid page for the selected paper.")
    elif page:
        raise HTTPException(400, "Select a paper before linking a page number.")

    note_id = uuid.uuid4().hex[:12]
    image_filename = ""
    if screenshot and screenshot.filename:
        mime = (screenshot.content_type or "").lower()
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise HTTPException(400, "Attach a PNG, JPEG, or WebP screenshot.")
        content = await screenshot.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise HTTPException(413, "Screenshot must be 8 MB or smaller.")
        valid_image = ((mime == "image/png" and content.startswith(b"\x89PNG\r\n\x1a\n")) or
                       (mime == "image/jpeg" and content.startswith(b"\xff\xd8\xff")) or
                       (mime == "image/webp" and content.startswith(b"RIFF") and content[8:12] == b"WEBP"))
        if not valid_image:
            raise HTTPException(400, "The attached file is not a valid supported image.")
        extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[mime]
        STICKY_IMAGES.mkdir(parents=True, exist_ok=True)
        image_filename = f"{note_id}{extension}"
        (STICKY_IMAGES / image_filename).write_bytes(content)

    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    note = {"id": note_id, "body": body[:4000], "color": color,
            "document_id": document_id or "", "page": page if document else 0,
            "image_filename": image_filename, "created_at": now, "updated_at": now}
    db["sticky_notes"].append(note)
    write_db(db)
    return public_sticky_note(note, db)


@app.put("/api/sticky-notes/{note_id}")
def update_sticky_note(note_id: str, req: StickyNoteUpdate):
    db = read_db()
    note = next((n for n in db["sticky_notes"] if n["id"] == note_id), None)
    if not note:
        raise HTTPException(404, "Sticky note not found")
    if not req.body.strip() and not note.get("image_filename"):
        raise HTTPException(400, "Add note text or keep a screenshot attached.")
    note.update({"body": req.body.strip(), "color": req.color,
                 "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
    write_db(db)
    return public_sticky_note(note, db)


@app.delete("/api/sticky-notes/{note_id}")
def delete_sticky_note(note_id: str):
    db = read_db()
    note = next((n for n in db["sticky_notes"] if n["id"] == note_id), None)
    if not note:
        raise HTTPException(404, "Sticky note not found")
    db["sticky_notes"] = [n for n in db["sticky_notes"] if n["id"] != note_id]
    if note.get("image_filename"):
        (STICKY_IMAGES / note["image_filename"]).unlink(missing_ok=True)
    write_db(db)
    return {"deleted": note_id}


@app.get("/api/sticky-notes/{note_id}/image")
def sticky_note_image(note_id: str):
    note = next((n for n in read_db()["sticky_notes"] if n["id"] == note_id), None)
    if not note or not note.get("image_filename"):
        raise HTTPException(404, "Screenshot not found")
    image_path = STICKY_IMAGES / note["image_filename"]
    if not image_path.exists():
        raise HTTPException(404, "Screenshot file is missing")
    return FileResponse(image_path, headers={"Cache-Control": "no-store"})


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
        if review["comparison_rows"][0].get("cells") is not None:
            headings = ["Criterion"] + [p["title"] for p in papers]
            out.extend(["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)])
        else:
            out.extend([f"| Criterion | {papers[0]['title']} | {papers[1]['title']} | Difference |",
                        "|---|---|---|---|"])
        for row in review["comparison_rows"]:
            values = row.get("cells", [row.get("left", ""), row.get("right", ""), row.get("difference", "")])
            cells = [str(value).replace("|", "\\|").replace("\n", " ")
                     for value in [row.get("criterion", ""), *values]]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    if review.get("claim_rows") or review.get("coverage_rows"):
        papers = review.get("comparison_papers", [])
        claim_mode = bool(review.get("claim_rows"))
        rows = review.get("claim_rows", []) if claim_mode else review.get("coverage_rows", [])
        first_heading = "Claim" if claim_mode else "Search question"
        headings = [first_heading] + [p.get("title", "Paper") for p in papers]
        out.extend(["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)])
        for row in rows:
            label = row.get("claim", row.get("question", ""))
            cells = [label]
            for cell in row.get("cells", []):
                if claim_mode:
                    check = cell.get("verification", {})
                    verified = (f"Human check: {check.get('status')}" +
                                (f" — {check.get('note')}" if check.get("note") else "")) if check else "Not human reviewed"
                    study = f"; Method/sample: {check.get('study_context')}" if check.get("study_context") else ""
                    cells.append(f"Model: {cell.get('status', 'unclear')}; {verified}{study}: {cell.get('text', '')} {' '.join(cell.get('refs', []))}")
                else:
                    cells.append(("Passage found " + str(cell.get("ref")) if cell.get("found") else "No match found"))
            out.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cells) + " |")
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
    if review.get("evidence"):
        out.extend(["## Cited source passages", ""])
        for i, item in enumerate(review["evidence"], start=1):
            out.append(f"- [S{i}] {item.get('paper', 'Source')} (p. {item.get('page', '?')}): {item.get('text', '').replace(chr(10), ' ')}")
        out.append("")
    return {"markdown": "\n".join(out)}
