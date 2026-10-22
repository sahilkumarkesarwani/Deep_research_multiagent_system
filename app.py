import html
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

# Guard: agents.py missing / broken dependencies
try:
    from agents import build_search_agent, build_scrap_agent, writer_chain, critic_chain
except Exception:
    st.error(
        "Could not import `agents.py`. Make sure it is in the same folder as `app.py` "
        "and that all its dependencies are installed in your venv."
    )
    st.code(traceback.format_exc())
    st.stop()

try:  # follow-up Q&A chain (add it to agents.py, see the snippet shown in the app)
    from agents import qa_chain
except ImportError:
    qa_chain = None

# ---------------------------------------------------------------- config
REQUIRED_KEYS = ["MISTRAL_API_KEY", "TAVILY_API_KEY"]  # edit to match your .env
MIN_LEN, MAX_LEN = 3, 300      # topic length limits
MAX_RETRIES = 2                # automatic retries per step (transient errors only)
MAX_HISTORY = 15               # runs kept in memory
STEP_KEYS = ["search_results", "scraped_content", "report", "critique"]
OPTIONAL_STEPS = {1, 3}        # pipeline continues if these fail (Read, Critique)
MAX_Q_LEN = 500                # follow-up question length limit
MAX_CONTEXT_CHARS = 12000      # research text sent with each follow-up
MAX_TURNS = 8                  # previous chat turns sent to the model

QA_SNIPPET = '''# add to agents.py (replace `llm` with the model variable you already use)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You answer follow-up questions about a research report. Base your answer on "
     "the context below. If the context does not cover the question, say so clearly "
     "instead of guessing. Be concise and use markdown.\\n\\nContext:\\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
qa_chain = qa_prompt | llm | StrOutputParser()
'''

STEPS = [
    ("🔎", "Search", "Finding recent sources"),
    ("🌐", "Read", "Scraping the best source"),
    ("✍️", "Write", "Drafting the report"),
    ("🧐", "Critique", "Reviewing quality"),
]

EXAMPLES = [
    "Solid-state battery breakthroughs",
    "State of open-source LLMs",
    "Impact of GLP-1 drugs on healthcare",
]

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero h1 {
    font-size: 2.4rem; font-weight: 800; margin-bottom: .1rem;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {opacity: .7; margin-top: 0; font-size: 1.05rem;}
.steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: 1rem 0;}
.step {
    border: 1px solid rgba(128,128,128,.3); border-radius: 12px;
    padding: .8rem 1rem; background: rgba(128,128,128,.06); transition: all .2s;
}
.step .ico {font-size: 1.4rem;}
.step .name {font-weight: 700; margin-top: .2rem;}
.step .desc {font-size: .8rem; opacity: .65;}
.step .time {font-size: .75rem; margin-top: .3rem; opacity: .8;}
.step.active {border-color: #6366f1; background: rgba(99,102,241,.14);
              box-shadow: 0 0 0 3px rgba(99,102,241,.18);}
.step.done {border-color: #22c55e; background: rgba(34,197,94,.10);}
.step.skipped {border-color: #f59e0b; background: rgba(245,158,11,.12);}
.step.failed {border-color: #ef4444; background: rgba(239,68,68,.12);}
.src {display:block; padding:.55rem .8rem; margin-bottom:.4rem; border-radius:8px;
      border:1px solid rgba(128,128,128,.25); text-decoration:none !important;
      word-break: break-all; font-size:.9rem;}
.src:hover {background: rgba(99,102,241,.12);}
@media (max-width: 700px) {.steps {grid-template-columns: repeat(2, 1fr);}}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def to_text(result) -> str:
    """Turn whatever an agent/chain returns into a plain string."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and "messages" in result:
        return to_text(result["messages"][-1])
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            return "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        return str(content)
    return str(result)


def extract_urls(*texts, limit=50) -> list[str]:
    seen, urls = set(), []
    for t in texts:
        for u in re.findall(r"https?://[^\s\)\]>\"'<]+", t or ""):
            u = u.rstrip(".,;:*")
            if u not in seen:
                seen.add(u)
                urls.append(u)
    return urls[:limit]


def extract_score(critique: str):
    m = re.search(r"(\d{1,2}(?:\.\d)?)\s*/\s*10", critique or "")
    if m and 0 <= float(m.group(1)) <= 10:
        return m.group(1)
    return None


def fmt_time(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 60 else f"{int(sec // 60)}m {int(sec % 60)}s"


def normalize_topic(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()


def validate_topic(t: str):
    """Return an error message, or None if the topic is fine / still empty."""
    t = normalize_topic(t)
    if not t:
        return None
    if len(t) < MIN_LEN:
        return f"Topic is too short (minimum {MIN_LEN} characters)."
    if not re.search(r"\w", t):
        return "Topic must contain letters or numbers."
    return None


def is_fatal(e: Exception) -> bool:
    """Errors that retrying cannot fix."""
    m = str(e).lower()
    return any(k in m for k in ("401", "unauthorized", "invalid api key",
                                "api key", "authentication", "permission"))


def friendly_error(e: Exception) -> str:
    m = str(e).lower()
    if is_fatal(e):
        return "Authentication failed. Check the API keys in your .env file."
    if "429" in m or "rate limit" in m or "quota" in m:
        return "Rate limit or quota reached. Wait a minute and retry."
    if isinstance(e, (TimeoutError, FuturesTimeout)) or "timed out" in m or "timeout" in m:
        return str(e) if "after" in m else "The step timed out. Try again or raise the timeout."
    if any(k in m for k in ("connection", "network", "dns", "unreachable", "ssl")):
        return "Network problem. Check your internet connection and retry."
    return str(e) or e.__class__.__name__


def call_with_timeout(fn, timeout):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except FuturesTimeout:
        raise TimeoutError(f"Step timed out after {timeout}s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def render_tracker(ph, current=None, timings=None, failed=None, skipped=()):
    timings = timings or {}
    cards = []
    for i, (ico, name, desc) in enumerate(STEPS):
        if failed == i:
            cls, extra = "failed", "failed"
        elif i in skipped:
            cls, extra = "skipped", "skipped"
        elif i in timings:
            cls, extra = "done", f"✓ {fmt_time(timings[i])}"
        elif current == i:
            cls, extra = "active", "working…"
        else:
            cls, extra = "", "waiting"
        cards.append(
            f'<div class="step {cls}"><div class="ico">{ico}</div>'
            f'<div class="name">{i + 1}. {name}</div>'
            f'<div class="desc">{desc}</div><div class="time">{extra}</div></div>'
        )
    ph.markdown(f'<div class="steps">{"".join(cards)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- pipeline
def new_state(topic):
    return {"topic": topic, "timings": {}, "warnings": {}, "error": None, "chat": []}


def run_step(i, state, scrape_limit):
    topic = state["topic"]
    if i == 0:
        res = build_search_agent().invoke(
            {"messages": [("user", f"Find recent and relevant information on the topic: {topic}")]}
        )
    elif i == 1:
        res = build_scrap_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:scrape_limit]}",
                    )
                ]
            }
        )
    elif i == 2:
        scraped = (state.get("scraped_content") or "").strip()
        scraped_part = scraped or "(unavailable - rely on the search results only)"
        combined = f"Search Results:\n{state['search_results']}\n\nScraped Content:\n{scraped_part}"
        res = writer_chain.invoke({"topic": topic, "research": combined})
    else:
        res = critic_chain.invoke({"report": state["report"]})
    return to_text(res)


def run_pipeline(state, start, scrape_limit, timeout, tracker_ph, progress, status_line):
    state["error"] = None
    for i in range(start, len(STEPS)):
        render_tracker(tracker_ph, current=i, timings=state["timings"],
                       skipped=set(state["warnings"]))
        status_line.caption(f"⏳ {STEPS[i][2]}…")
        progress.progress(i / len(STEPS))

        t0 = time.time()
        out, last_err = None, None
        for attempt in range(MAX_RETRIES + 1):
            try:
                out = call_with_timeout(lambda: run_step(i, state, scrape_limit), timeout)
                if not out or not out.strip():
                    raise ValueError("The agent returned an empty response.")
                break
            except Exception as e:
                out, last_err = None, e
                if is_fatal(e) or attempt == MAX_RETRIES:
                    break
                status_line.caption(
                    f"⚠️ {STEPS[i][1]} failed, retrying ({attempt + 2}/{MAX_RETRIES + 1})…"
                )
                time.sleep(2 * (attempt + 1))

        elapsed = time.time() - t0
        if out is not None:
            state[STEP_KEYS[i]] = out
            state["timings"][i] = elapsed
        elif i in OPTIONAL_STEPS:
            # graceful degradation: keep going without this step
            state[STEP_KEYS[i]] = ""
            state["timings"][i] = elapsed
            state["warnings"][i] = f"{STEPS[i][1]} step skipped: {friendly_error(last_err)}"
        else:
            state["error"] = {
                "step": i,
                "msg": friendly_error(last_err),
                "trace": "".join(traceback.format_exception(last_err)),
            }
            render_tracker(tracker_ph, timings=state["timings"], failed=i,
                           skipped=set(state["warnings"]))
            return state

    state["total_time"] = sum(state["timings"].values())
    state["finished"] = datetime.now().strftime("%H:%M")
    render_tracker(tracker_ph, timings=state["timings"], skipped=set(state["warnings"]))
    return state


# ---------------------------------------------------------------- follow-up Q&A
def build_context(res, extra=""):
    parts = [f"# Report\n{res.get('report', '')}"]
    if extra:
        parts.append(f"# Fresh web search for this question\n{extra}")
    if res.get("search_results"):
        parts.append(f"# Search results\n{res['search_results']}")
    if res.get("scraped_content"):
        parts.append(f"# Scraped content\n{res['scraped_content']}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def history_messages(chat):
    return [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in chat[-MAX_TURNS * 2:]
    ]


def answer_question(res, question, use_web, timeout):
    extra = ""
    if use_web:
        try:
            extra = call_with_timeout(
                lambda: to_text(build_search_agent().invoke({"messages": [(
                    "user",
                    f"Find information to answer this question about '{res['topic']}': {question}",
                )]})),
                timeout,
            )
        except Exception:
            extra = ""  # fall back to the existing research only

    payload = {
        "context": build_context(res, extra),
        "history": history_messages(res["chat"]),
        "question": question,
    }
    last_err = None
    for attempt in range(2):  # one retry for transient errors
        try:
            out = call_with_timeout(lambda: to_text(qa_chain.invoke(payload)), timeout)
            if not out.strip():
                raise ValueError("The model returned an empty answer.")
            return out
        except Exception as e:
            last_err = e
            if is_fatal(e):
                break
            time.sleep(2)
    raise last_err


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("current", None)
ss.setdefault("topic_input", "")
ss.setdefault("running", False)

# A job is created by a button callback; consume it exactly once.
job = ss.pop("job", None)
if ss.running and not job:  # stale flag from an interrupted run
    ss.running = False
running = ss.running
if ss.current is not None and ss.current >= len(ss.history):
    ss.current = None


def set_example(text):
    ss.topic_input = text


def start_run():
    ss.job = {"topic": normalize_topic(ss.topic_input)}
    ss.running = True


def start_retry(idx):
    ss.job = {"retry_idx": idx}
    ss.running = True


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    scrape_limit = st.slider(
        "Search text passed to reader (chars)",
        500, 5000, 2000, 250,
        help="Your original code used 800, which can cut URLs in half.",
        disabled=running,
    )
    timeout = st.slider("Timeout per step (seconds)", 30, 600, 180, 30, disabled=running)
    show_raw = st.toggle("Show raw agent output", value=False, disabled=running)
    use_web = st.toggle(
        "Web search for follow-ups", value=False, disabled=running,
        help="Off: answers come only from the research already done. "
             "On: the search agent also looks up fresh info for each question (slower).",
    )

    st.header("🕘 History")
    if not ss.history:
        st.caption("No runs yet.")
    for idx, h in enumerate(reversed(ss.history)):
        real_idx = len(ss.history) - 1 - idx
        icon = "❌ " if h["error"] else ("⚠️ " if h["warnings"] else "")
        label = f"{icon}{h['topic'][:34]}{'…' if len(h['topic']) > 34 else ''}"
        if st.button(label, key=f"hist_{real_idx}", use_container_width=True,
                     disabled=running,
                     type="primary" if real_idx == ss.current else "secondary"):
            ss.current = real_idx
            st.rerun()
    if ss.history and st.button("🗑️ Clear history", use_container_width=True, disabled=running):
        ss.history = []
        ss.current = None
        st.rerun()

# ---------------------------------------------------------------- header + input
st.markdown(
    '<div class="hero"><h1>Multi-Agent Research</h1>'
    "<p>Search, read, write and critique, handled by a team of AI agents.</p></div>",
    unsafe_allow_html=True,
)

missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    st.warning(
        f"Missing environment variable(s): {', '.join(missing)}. "
        "Add them to your `.env` file and restart the app. "
        "(If you use different providers, edit `REQUIRED_KEYS` at the top of `app.py`.)"
    )

c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
with c1:
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        max_chars=MAX_LEN,
        placeholder="e.g. Latest advances in solid-state batteries",
        disabled=running,
    )
topic_error = validate_topic(topic)
with c2:
    st.button(
        "🚀 Run",
        type="primary",
        use_container_width=True,
        on_click=start_run,
        disabled=running or not normalize_topic(topic) or bool(topic_error),
    )
if topic_error:
    st.caption(f"⚠️ {topic_error}")

ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=set_example, args=(ex,),
               use_container_width=True, disabled=running)

tracker_ph = st.empty()
progress = st.empty()
status_line = st.empty()

# ---------------------------------------------------------------- run
if job:
    try:
        if "retry_idx" in job:
            idx = job["retry_idx"]
            state = ss.history[idx]
            start = state["error"]["step"] if state.get("error") else 0
            for k in range(start, len(STEPS)):
                state["timings"].pop(k, None)
                state["warnings"].pop(k, None)
            is_new = False
        else:
            state, start, is_new = new_state(job["topic"]), 0, True

        prog = progress.progress(0.0)
        render_tracker(tracker_ph, timings=state["timings"], skipped=set(state["warnings"]))
        run_pipeline(state, start, scrape_limit, timeout, tracker_ph, prog, status_line)

        if is_new:
            ss.history.append(state)
            while len(ss.history) > MAX_HISTORY:
                ss.history.pop(0)
            ss.current = len(ss.history) - 1
        else:
            ss.current = idx
        ss.just_finished = "error" if state["error"] else "ok"
    finally:
        ss.running = False
    st.rerun()  # re-enable widgets and show results

elif ss.current is not None:
    cur = ss.history[ss.current]
    render_tracker(
        tracker_ph,
        timings=cur["timings"],
        failed=cur["error"]["step"] if cur["error"] else None,
        skipped=set(cur["warnings"]),
    )
    flash = ss.pop("just_finished", None)
    if flash == "ok":
        progress.progress(1.0)
        status_line.caption("✅ Done")
    elif flash == "error":
        status_line.caption("❌ Stopped because of an error")

# ---------------------------------------------------------------- results
if ss.current is not None:
    res = ss.history[ss.current]

    if res["error"]:
        err = res["error"]
        st.error(f"Step {err['step'] + 1} ({STEPS[err['step']][1]}) failed: {err['msg']}")
        st.button("🔁 Retry from failed step", on_click=start_retry, args=(ss.current,),
                  disabled=running)
        with st.expander("Error details"):
            st.code(err["trace"])
    for msg in res["warnings"].values():
        st.warning(msg)

    report = res.get("report", "")
    critique = res.get("critique", "")
    urls = extract_urls(res.get("search_results"), res.get("scraped_content"))
    score = extract_score(critique)

    st.divider()
    st.subheader(f"📌 {res['topic']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Report words", f"{len(report.split()):,}" if report else "—")
    m2.metric("Sources found", len(urls) if urls else "—")
    m3.metric("Critic score", f"{score}/10" if score else "—")
    m4.metric("Total time", fmt_time(res["total_time"]) if "total_time" in res else "—")

    tab_report, tab_critique, tab_sources, tab_raw = st.tabs(
        ["📄 Report", "🧐 Critique", "🔗 Sources", "🧪 Raw output"]
    )

    with tab_report:
        if report:
            st.markdown(report)
            st.divider()
            d1, d2 = st.columns(2)
            safe = re.sub(r"[^a-zA-Z0-9]+", "_", res["topic"])[:40].strip("_") or "report"
            d1.download_button("⬇️ Download report (.md)", report, f"{safe}.md",
                               "text/markdown", use_container_width=True)
            full = f"{report}\n\n---\n\n## Critique\n\n{critique or '_No critique available._'}"
            d2.download_button("⬇️ Report + critique (.md)", full, f"{safe}_full.md",
                               "text/markdown", use_container_width=True)
            with st.expander("📋 Copy as plain text"):
                st.code(report, language="markdown")
        else:
            st.info("No report was generated.")

    with tab_critique:
        st.markdown(critique if critique else "_No critique available._")

    with tab_sources:
        if urls:
            for u in urls:
                safe_u = html.escape(u, quote=True)
                st.markdown(
                    f'<a class="src" href="{safe_u}" target="_blank" '
                    f'rel="noopener noreferrer">🔗 {safe_u}</a>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No URLs were found in the agent output.")

    with tab_raw:
        if show_raw:
            with st.expander("Search agent output", expanded=False):
                st.markdown(res.get("search_results") or "—")
            with st.expander("Scraped content", expanded=False):
                st.markdown(res.get("scraped_content") or "—")
        else:
            st.caption("Enable **Show raw agent output** in the sidebar to see this.")

    # ------------------------------------------------------------ follow-up Q&A
    if report:
        res.setdefault("chat", [])
        st.divider()
        h1, h2 = st.columns([5, 1], vertical_alignment="center")
        h1.subheader("💬 Ask about this research")
        if res["chat"] and h2.button("🧹 Clear chat", use_container_width=True, disabled=running):
            res["chat"] = []
            st.rerun()

        if qa_chain is None:
            st.info("The follow-up chat needs a `qa_chain` in your `agents.py`. Add this and restart:")
            st.code(QA_SNIPPET, language="python")
        else:
            if not res["chat"]:
                st.caption("Have a doubt about the report? Ask below. Answers are based on this research.")
            for m in res["chat"]:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            question = st.chat_input("Ask a follow-up question…", max_chars=MAX_Q_LEN,
                                     disabled=running)
            q = normalize_topic(question) if question else ""
            if q:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Thinking…"):
                            ans = answer_question(res, q, use_web, timeout)
                        st.markdown(ans)
                        res["chat"] += [{"role": "user", "content": q},
                                        {"role": "assistant", "content": ans}]
                    except Exception as e:
                        st.error(f"Couldn't answer that: {friendly_error(e)}")