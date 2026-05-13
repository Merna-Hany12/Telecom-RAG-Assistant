from __future__ import annotations

import os
import re
import json
import numpy as np
import faiss
import unicodedata
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

load_dotenv()

DATA_PATH       = r"C:\Users\merna\Desktop\telecom-rag-assistant\data\data"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# ── Cache file paths ────────────────────────────────────────
CACHE_DIR     = os.path.join(os.path.dirname(__file__), "cache")
INDEX_PATH    = os.path.join(CACHE_DIR, "faiss.index")
CHUNKS_PATH   = os.path.join(CACHE_DIR, "chunks.json")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.json")

# ── Conversational memory settings ─────────────────────────
MAX_HISTORY_TURNS = 6   # keep last 6 exchanges (12 messages) per session


class RagCore:
    """
    Full NileTel RAG pipeline with per-session conversational memory.

    Fixes applied (v2):
    ─────────────────────────────────────────────────────────
    FIX 1 — Memory content bug:
        _generate_answer used to save the full context blob (retrieved docs +
        query) as the user's history entry.  Now only the clean `query` string
        is stored, so the history that gets replayed to the LLM is realistic
        conversation turns, not giant walls of retrieval context.

    FIX 2 — Ticket carries problem context:
        When the user says "ارفع تذكرة" the router used to return a hardcoded
        reply with zero problem detail.  Now the handler walks backwards
        through the session history, finds the last user message that is NOT
        itself a ticket request, and injects that as "المشكلة المسجلة" in both
        the reply and the returned metadata.  The ticket details are also
        exposed as `ticket_problem` in the response dict so the API / front-end
        can store/log them.

    FIX 3 — _route_query receives session_id:
        Signature updated so the ticket handler can read history without a
        second lookup.

    Memory is stored in self.sessions[session_id] as a list of
    {"role": "user"|"assistant", "content": "..."} dicts — exactly
    the format Groq expects — and trimmed to MAX_HISTORY_TURNS pairs.

    First run  → builds embeddings & FAISS index, saves to /cache  (~50 sec)
    Later runs → loads from /cache, skips embedding step            (~2 sec)

    If you update your .md files, call rag.clear_cache() then restart.
    """

    GREETING_KEYWORDS: List[str] = [
        "ازيك", "ازيك؟", "ازيكو", "عامل إيه", "عامل ايه",
        "مرحبا", "مرحباً", "اهلا", "أهلا", "اهلاً",
        "السلام عليكم", "سلام", "صباح الخير", "مساء الخير",
        "hello", "hi", "hey", "good morning", "good evening",
        "كيف حالك", "كيفك", "شو اخبارك",
    ]

    TICKET_KEYWORDS: List[str] = [
        "تذكرة", "تذكره", "اعمل تذكرة", "ارفع تذكرة", "افتح تذكرة",
        "فتح تذكرة", "رفع تذكرة", "عمل تذكرة",
        "مهندس", "ابعت مهندس", "ارسل مهندس", "محتاج مهندس",
        "فريق صيانة", "زيارة ميدانية", "كشف تقني",
        "تصعيد", "تصعيد المشكلة", "مسؤول", "مشرف", "طلب دعم",
        "ارفع شكوى", "شكوى رسمية", "complaint",
        "سجل", "سجل مشكلة", "بلغ", "بلغ المشكلة",
    ]

    TELECOM_KEYWORDS: List[str] = [
        "نت", "انترنت", "اتصال", "شبكة", "سيگنال", "سيجنال",
        "5g", "4g", "3g", "خدمة", "باقة", "روتر", "راوتر",
        "سرعة", "بطء", "مقطوع", "انقطاع", "فاتورة", "رصيد",
        "شحن", "تجديد", "اشتراك", "كابل", "fiber", "فايبر",
        "ارسال", "استقبال", "تغطية", "مكالمة", "sms", "رسالة",
        "نايل تل", "niletel", "الشركة", "دعم فني", "خدمة عملاء",
        "عطل", "مشكلة", "بلاغ", "wifi", "واي فاي",
        "throttling", "latency", "ping", "modem",
    ]

    OUT_OF_SCOPE_KEYWORDS: List[str] = [
        "فيلم", "مسلسل", "افلام", "سينما", "مسلسلات",
        "اغنية", "اغاني", "موسيقى", "كونسرت",
        "اكل", "مطعم", "وصفة", "طبخ", "أكل",
        "رياضة", "كورة", "مباراة", "فريق", "دوري","ماتش",
        "كرة القدم", "اولمبياد",
        "سياسة", "انتخابات", "حكومة", "وزير", "برلمان",
        "بورصة", "اسهم", "عملة", "دولار", "عقار",
        "طقس", "جو", "حب", "علاقة", "صحة", "طب", "دواء",
    ]

    # ───────────────────────────────────────────────────────────
    def __init__(self, data_path: str = DATA_PATH) -> None:
        print("[RagCore] Initializing...")
        os.makedirs(CACHE_DIR, exist_ok=True)

        # ── Per-session conversation history ────────────────
        # { session_id: [{"role": "user"|"assistant", "content": "..."}, ...] }
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

        cache_exists = (
            os.path.exists(INDEX_PATH)
            and os.path.exists(CHUNKS_PATH)
            and os.path.exists(METADATA_PATH)
        )

        if cache_exists:
            print("[RagCore] ✅ Cache found — loading from disk...")
            self.index, self.all_chunks, self.metadata = self._load_cache()
            print("[RagCore] Loading embedding model for query encoding...")
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
            print("[RagCore] Building BM25 index from cached chunks...")
            tokenized_chunks = [chunk.split() for chunk in self.all_chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
        else:
            print("[RagCore] No cache — building from scratch (first run only)...")
            self.all_chunks: List[str] = []
            self.metadata:   List[Dict[str, str]] = []

            for file in os.listdir(data_path):
                if file.endswith(".md"):
                    with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                        text = f.read()
                    for chunk in self._chunk_text(text):
                        self.all_chunks.append(chunk)
                        self.metadata.append({"source": file})

            print(f"[RagCore] Loaded {len(self.all_chunks)} chunks from {data_path}")

            self.embed_model, embeddings = self._create_embeddings(self.all_chunks)
            self.index = self._build_faiss_index(embeddings)
            print("[RagCore] Building BM25 index...")
            tokenized_chunks = [chunk.split() for chunk in self.all_chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            self._save_cache(embeddings)

        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[RagCore] Ready ✅")

    # ══════════════════════════════════════════════════════════
    # MEMORY HELPERS
    # ══════════════════════════════════════════════════════════

    def _get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Returns the sliding-window history for a session."""
        return self.sessions.get(session_id, [])

    def _append_history(self, session_id: str, role: str, content: str) -> None:
        """Appends a message and trims to MAX_HISTORY_TURNS pairs."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        # Keep only the last MAX_HISTORY_TURNS * 2 messages (user + assistant pairs)
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.sessions[session_id]) > max_msgs:
            self.sessions[session_id] = self.sessions[session_id][-max_msgs:]

    def clear_session(self, session_id: str) -> None:
        """Wipes memory for a specific session (called when user clicks 'مسح المحادثة')."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"[RagCore] Session {session_id} cleared.")

    # ══════════════════════════════════════════════════════════
    # TICKET PROBLEM EXTRACTION  (FIX 2)
    # ══════════════════════════════════════════════════════════

    # Telecom problem indicators — presence of any of these means the text
    # is a real problem description, not just a status confirmation.
    PROBLEM_INDICATORS: List[str] = [
        "نت", "انترنت", "شبكة", "سرعة", "بطء", "بطيء", "مقطوع", "انقطاع",
        "واي فاي", "wifi", "راوتر", "روتر", "فايبر", "fiber", "كابل",
        "فاتورة", "رصيد", "باقة", "اشتراك", "مكالمة", "سيجنال", "تغطية",
        "عطل", "مشكلة", "خراب", "وقع", "اتقطع", "بيوقع",
    ]

    def _describes_problem(self, text: str) -> bool:
        """True if text contains at least one telecom problem indicator keyword."""
        t = text.lower()
        return any(kw in t for kw in self.PROBLEM_INDICATORS)

    def _strip_ticket_keywords(self, text: str) -> str:
        """Remove all ticket/escalation keywords from text and clean up."""
        for kw in self.TICKET_KEYWORDS:
            text = text.replace(kw, "")
        return text.strip(" ،,.-\n")

    def _extract_problem_from_history(self, session_id: str, current_query: str) -> str:
        """
        History-first strategy:

        Step 1 — Walk history backwards.
                  Find the most recent user message that:
                    (a) contains no ticket keywords, AND
                    (b) contains at least one telecom problem indicator.
                  This is almost always the message where the user first
                  described their issue in detail.

        Step 2 — Inline detection (current query only).
                  Only use current_query as the problem source when the user
                  wrote both problem AND ticket request in ONE message, e.g.:
                      "النت بطيء اعملي تذكرة"
                  Detected by: stripped current query still has a problem indicator.

        Step 3 — Fallback to current_query as-is.

        This guarantees that follow-up messages like:
            "مش شغال برده بطئ لسه طيب ابعت مهندس"
        never override the original detailed problem report sitting in history.
        """
        history = self._get_history(session_id)

        # ── Step 1: history OLDEST-FIRST ─────────────────────────────────────
        # We want the FIRST message where the user described their problem,
        # not the most recent follow-up like "لا لسه بطئ برده".
        for msg in history:
            if msg["role"] != "user":
                continue
            msg_text = msg["content"].strip()

            # Skip if it's a ticket request
            has_ticket_kw = any(kw in msg_text.lower() for kw in self.TICKET_KEYWORDS)
            if has_ticket_kw:
                continue

            # Must describe an actual telecom problem
            if self._describes_problem(msg_text):
                print(f"[RagCore] ✅ Ticket problem from history (oldest): {msg_text}")
                return msg_text

        # ── Step 2: inline problem + ticket in same message ───────────────────
        stripped = self._strip_ticket_keywords(current_query)
        if stripped and self._describes_problem(stripped):
            print(f"[RagCore] ✅ Ticket problem inline (current): {stripped}")
            return stripped

        # ── Step 3: fallback ──────────────────────────────────────────────────
        print(f"[RagCore] ⚠️ Ticket problem fallback: {current_query}")
        return current_query

    # ══════════════════════════════════════════════════════════
    # CACHE HELPERS
    # ══════════════════════════════════════════════════════════

    def _save_cache(self, embeddings: np.ndarray) -> None:
        print(f"[RagCore] Saving cache to {CACHE_DIR} ...")
        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH,   "w", encoding="utf-8") as f:
            json.dump(self.all_chunks, f, ensure_ascii=False, indent=2)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata,   f, ensure_ascii=False, indent=2)
        print("[RagCore] Cache saved ✅")

    def _load_cache(self) -> Tuple[faiss.IndexFlatIP, List[str], List[Dict]]:
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH,   "r", encoding="utf-8") as f:
            chunks = json.load(f)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"[RagCore] Loaded {index.ntotal} vectors from cache")
        return index, chunks, metadata

    def clear_cache(self) -> None:
        for path in [INDEX_PATH, CHUNKS_PATH, METADATA_PATH]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[RagCore] Deleted {path}")
        print("[RagCore] Cache cleared — will rebuild on next startup.")

    # ══════════════════════════════════════════════════════════
    # PUBLIC METHOD
    # ══════════════════════════════════════════════════════════

    def run_rag_pipeline(self, query: str, session_id: str = "default") -> Dict[str, object]:
        print(f"\n{'='*80}")
        print(f"[RagCore] Session: {session_id}  Query: {query}")

        # FIX 3 — pass session_id into router so ticket handler can use history
        route = self._route_query(query, session_id)
        print(f"[RagCore] Route: {route}")

        # ── Greeting ──────────────────────────────────────────
        if route == "greeting":
            reply = "أهلاً وسهلاً يا فندم! أنا مساعد NileTel، إزيك؟ بإيه أقدر أساعدك النهارده؟"
            self._append_history(session_id, "user",      query)
            self._append_history(session_id, "assistant", reply)
            return self._make_response(answer=reply, needs_action="NO", sources=[])

        # ── Out of scope ──────────────────────────────────────
        if route == "out_of_scope":
            reply = "آسف يا فندم، مش هقدر أساعدك في الموضوع ده. أنا متخصص في دعم عملاء NileTel."
            self._append_history(session_id, "user",      query)
            self._append_history(session_id, "assistant", reply)
            return self._make_response(answer=reply, needs_action="NO", sources=[])

        # ── Ticket  (FIX 2) ───────────────────────────────────
        if route == "ticket":
            problem = self._extract_problem_from_history(session_id, query)
            problem_display = self._fix_bidi_text(problem)

            reply = (
                f"تمام يا فندم، هبدأ في إنشاء التذكرة حالاً.\n"
                f"المشكلة المسجلة: {problem_display}\n"
                f"مهندس الدعم الفني هيتواصل مع حضرتك قريباً."
            )
            self._append_history(session_id, "user",      query)
            self._append_history(session_id, "assistant", reply)

            response = self._make_response(answer=reply, needs_action="YES", sources=[])
            response["ticket_problem"] = problem   # expose raw problem for API / logging
            return response

        # ── Normal RAG flow ───────────────────────────────────
        cleaned = self._clean_query(query)
        results = self._retrieve(cleaned, top_k=6)
        return self._generate_answer(query=query, cleaned_query=cleaned,
                                     retrieved_results=results, session_id=session_id)

    # ══════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        CHUNK_SIZE = 10
        STEP = 7
        chunks = []
        i = 0
        while True:
            window = sentences[i:i + CHUNK_SIZE]
            if not window:
                break
            chunks.append(" ".join(window))
            i += STEP
        return chunks

    def _create_embeddings(self, chunks: List[str]) -> Tuple[SentenceTransformer, np.ndarray]:
        print("[RagCore] Creating embeddings...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        print(f"[RagCore] Embedding shape: {embeddings.shape}")
        return model, embeddings

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        print(f"[RagCore] FAISS index built — {index.ntotal} vectors")
        return index

    def _retrieve_semantic(self, query: str, top_k: int = 6) -> List[Dict[str, object]]:
        query_emb = self.embed_model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        results: List[Dict[str, object]] = []
        for idx, score in zip(indices[0], distances[0]):
            if score > 0.4:
                results.append({
                    "idx":    int(idx),
                    "text":   self.all_chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score":  float(score),
                })
        print(f"[RagCore] Semantic results: {len(results)}")
        return results

    def _retrieve_bm25(self, query: str, top_k: int = 6) -> List[Dict[str, object]]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, object]] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "idx":    int(idx),
                    "text":   self.all_chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score":  float(scores[idx]),
                })
        print(f"[RagCore] BM25 results: {len(results)}")
        return results

    def _rrf_fusion(self, semantic_results: List[Dict], bm25_results: List[Dict],
                    k: int = 60, top_k: int = 6) -> List[Dict[str, object]]:
        rrf_scores: Dict[int, float] = {}
        lookup:     Dict[int, Dict]  = {}

        for rank, result in enumerate(semantic_results):
            idx = result["idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank)
            lookup[idx] = result

        for rank, result in enumerate(bm25_results):
            idx = result["idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank)
            if idx not in lookup:
                lookup[idx] = result

        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

        fused: List[Dict[str, object]] = []
        for idx in sorted_indices[:top_k]:
            entry = lookup[idx].copy()
            entry["score"] = rrf_scores[idx]
            fused.append(entry)

        print(f"[RagCore] RRF fused results: {len(fused)}")
        return fused

    def _retrieve(self, query: str, top_k: int = 6) -> List[Dict[str, object]]:
        semantic = self._retrieve_semantic(query, top_k)
        bm25     = self._retrieve_bm25(query, top_k)
        return self._rrf_fusion(semantic, bm25, top_k=top_k)

    def _route_query(self, query: str, session_id: str = "default") -> str:
        q = query.strip().lower()

        EXPLICIT_TICKET_TRIGGERS: List[str] = [
            "تذكرة", "تذكره",
            "ابعت مهندس", "ارسل مهندس", "محتاج مهندس",
            "فريق صيانة", "زيارة ميدانية", "كشف تقني",
            "ارفع شكوى", "شكوى رسمية", "complaint",
            "تصعيد المشكلة", "سجل مشكلة", "بلغ المشكلة",
        ]

        def exact_phrase_match(text: str, phrases: List[str]) -> bool:
            for phrase in phrases:
                if " " in phrase:
                    if phrase in text:
                        return True
                else:
                    pattern = r'(^|\s)' + re.escape(phrase) + r'(\s|$)'
                    if re.search(pattern, text):
                        return True
            return False

        if exact_phrase_match(q, EXPLICIT_TICKET_TRIGGERS):
            return "ticket"

        if any(kw in q for kw in self.GREETING_KEYWORDS):
            return "greeting"
        if any(kw in q for kw in self.OUT_OF_SCOPE_KEYWORDS):
            return "out_of_scope"

        return "chat"

    def _clean_query(self, query: str) -> str:
        for kw in self.GREETING_KEYWORDS:
            query = query.replace(kw, "")
        return query.strip()

    @staticmethod
    def _fix_bidi_text(text: str) -> str:
        """Wraps English words inside Arabic text with LTR/RTL marks."""
        LTR_MARK = "\u200E"
        RTL_MARK = "\u200F"
        result = re.sub(
            r'([A-Za-z0-9][A-Za-z0-9\s\.\-\/\%\_]*[A-Za-z0-9]|[A-Za-z0-9])',
            lambda m: f"{LTR_MARK}{m.group()}{RTL_MARK}",
            text,
        )
        return result

    def _generate_answer(
        self,
        query: str,                          # original user query  (for history)
        cleaned_query: str,                  # greeting-stripped query (for retrieval prompt)
        retrieved_results: List[Dict[str, object]],
        session_id: str = "default",
    ) -> Dict[str, object]:

        if not retrieved_results:
            reply = "مش متأكد من البيانات المتاحة يا فندم."
            # FIX 1 — save clean query, not context blob
            self._append_history(session_id, "user",      query)
            self._append_history(session_id, "assistant", reply)
            return self._make_response(answer=reply, needs_action="NO", sources=[])

        context = "\n\n".join([
            f"Source: {res['source']}\n{res['text']}" for res in retrieved_results
        ])

        system_prompt = """أنت مساعد دعم عملاء محترف في شركة NileTel للاتصالات.

قواعد صارمة يجب اتباعها:
- أجب باللهجة المصرية الطبيعية وبلباقة (يا فندم، تمام، هنحلها...)
- استخدم فقط المعلومات الموجودة في السياق. ممنوع التأليف.
- لا تختلق أرقام تذاكر أو تفاصيل وهمية.
- عند ذكر مصطلحات إنجليزية (مثل 5G, WiFi, fiber) اكتبها في منتصف الجملة العربية مع مسافة قبلها وبعدها.
- لا تبدأ جملة بكلمة إنجليزية — ابدأ دائماً بكلمة عربية.
- إذا كان المستخدم يكمّل سؤالاً سابقاً أو يقول "وإيه تاني" أو "زِد" — استخدم سياق المحادثة السابقة.
needs_action = YES فقط إذا المستخدم كتب طلب صريح مثل:
"اعمل تذكرة" أو "ابعت مهندس"
غير ذلك ALWAYS NO

تنسيق الإجابة (إلزامي):
answer: <نص إجابتك هنا>
needs_action: <YES أو NO فقط>

"""

        # ── Build messages: system + sliding history + new user turn ──────────
        # FIX 1 — history already contains clean queries (not context blobs),
        #          so replaying it is coherent for the LLM.
        history = self._get_history(session_id)

        # The retrieval context goes only in the current turn's content, NOT in history
        user_content = (
            f"السياق المتاح (استخدمه فقط):\n{context}\n\n"
            f"السؤال: {cleaned_query}"
        )

        messages = (
            [{"role": "system", "content": system_prompt}]
            + history                                           # past turns (clean text only)
            + [{"role": "user", "content": user_content}]      # current turn with context
        )

        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )

        raw_text = response.choices[0].message.content.strip()

        answer_match = re.search(
            r'answer\s*:\s*(.+?)(?=needs_action\s*:|$)', raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        clean_answer = answer_match.group(1).strip() if answer_match else raw_text

        action_match = re.search(r'needs_action\s*:\s*(YES|NO)', raw_text, flags=re.IGNORECASE)
        needs_action = action_match.group(1).upper() if action_match else "NO"

        # FIX 1 — save original query (not the big user_content with context)
        self._append_history(session_id, "user",      query)
        self._append_history(session_id, "assistant", clean_answer)

        sources: List[str] = [str(res["source"]) for res in retrieved_results]
        return self._make_response(answer=clean_answer, needs_action=needs_action, sources=sources)

    @staticmethod
    def _make_response(answer: str, needs_action: str, sources: List[str]) -> Dict[str, object]:
        unique_sources   = list(dict.fromkeys(sources))
        displayed_source = unique_sources[0] if unique_sources else ""
        clean_answer     = RagCore._fix_bidi_text(answer)
        return {
            "answer":           clean_answer,
            "needs_action":     needs_action,
            "sources":          unique_sources,
            "displayed_source": displayed_source,
            "ticket_problem":   "",    # filled by ticket route only
        }