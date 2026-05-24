import streamlit as st
import time
import os
from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
custom_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1200px;
}

/* HERO */
.hero {
    text-align: center;
    padding: 3rem 0;
}

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #ff8c32;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    color: #f0ebe0;
    margin-bottom: 1rem;
}

.hero h1 span {
    color: #ff8c32;
}

.hero-sub {
    color: #a09890;
    font-size: 1rem;
    max-width: 700px;
    margin: auto;
    line-height: 1.7;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,140,50,0.4),
        transparent
    );
    margin: 2rem 0;
}

/* INPUT CARD */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,140,50,0.15);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(8px);
}

/* INPUT */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    color: #f0ebe0 !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput > label {
    color: #ff8c32 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.1em !important;
    font-size: 0.75rem !important;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%);
    color: #0a0a0f;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
    padding: 0.8rem;
    font-size: 1rem;
}

/* SECTION */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

/* STEP CARD */
.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

.step-title {
    font-weight: 700;
    color: #f0ebe0;
}

.step-status {
    font-size: 0.8rem;
    color: #ff8c32;
    margin-top: 0.3rem;
}

/* RESULT PANELS */
.result-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.notice {
    text-align: center;
    color: #605850;
    font-size: 0.75rem;
    margin-top: 3rem;
    font-family: 'DM Mono', monospace;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}

if "running" not in st.session_state:
    st.session_state.running = False

if "done" not in st.session_state:
    st.session_state.done = False

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>

    <p class="hero-sub">
        AI-powered multi-agent research assistant using
        LangChain, Tavily, Mistral and Streamlit.
    </p>
</div>

<div class="divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 4])

# ─────────────────────────────────────────────────────────────
# LEFT SIDE
# ─────────────────────────────────────────────────────────────
with col1:

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Future of quantum computing",
        key="topic_input"
    )

    run_btn = st.button(
        "⚡ Run Research Pipeline",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RIGHT SIDE
# ─────────────────────────────────────────────────────────────
with col2:

    st.markdown(
        '<div class="section-heading">Pipeline</div>',
        unsafe_allow_html=True
    )

    steps = [
        "Search Agent",
        "Reader Agent",
        "Writer Chain",
        "Critic Chain"
    ]

    for step in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">{step}</div>
            <div class="step-status">READY</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────
if run_btn:

    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:

        st.session_state.running = True
        st.session_state.done = False
        st.session_state.results = {}

        try:

            results = {}

            # SEARCH AGENT
            with st.spinner("🔍 Search Agent working..."):

                search_agent = build_search_agent()

                sr = search_agent.invoke({
                    "messages": [
                        (
                            "user",
                            f"Find detailed information about {topic}"
                        )
                    ]
                })

                results["search"] = sr["messages"][-1].content

            # READER AGENT
            with st.spinner("📄 Reader Agent scraping content..."):

                reader_agent = build_reader_agent()

                rr = reader_agent.invoke({
                    "messages": [
                        (
                            "user",
                            f"""
                            Based on the following search results,
                            scrape and extract useful information.

                            SEARCH RESULTS:
                            {results['search']}
                            """
                        )
                    ]
                })

                results["reader"] = rr["messages"][-1].content

            # WRITER
            with st.spinner("✍️ Writing report..."):

                combined = f"""
                SEARCH RESULTS:
                {results['search']}

                SCRAPED CONTENT:
                {results['reader']}
                """

                report = writer_chain.invoke({
                    "topic": topic,
                    "research": combined
                })

                results["writer"] = report

            # CRITIC
            with st.spinner("🧐 Reviewing report..."):

                critic = critic_chain.invoke({
                    "report": results["writer"]
                })

                results["critic"] = critic

            st.session_state.results = results
            st.session_state.done = True
            st.session_state.running = False

        except Exception as e:

            st.error(f"Error: {str(e)}")
            st.session_state.running = False

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
r = st.session_state.results

if r:

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Results</div>',
        unsafe_allow_html=True
    )

    # SEARCH RESULTS
    if "search" in r:
        with st.expander("🔍 Search Results"):
            st.write(r["search"])

    # READER RESULTS
    if "reader" in r:
        with st.expander("📄 Reader Output"):
            st.write(r["reader"])

    # FINAL REPORT
    if "writer" in r:

        st.markdown("""
        <div class="result-panel">
        """, unsafe_allow_html=True)

        st.subheader("📝 Final Research Report")

        st.markdown(
            r["writer"],
            unsafe_allow_html=False
        )

        st.download_button(
            label="⬇ Download Report",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # CRITIC
    if "critic" in r:

        st.markdown("""
        <div class="result-panel">
        """, unsafe_allow_html=True)

        st.subheader("🧐 Critic Feedback")

        st.markdown(
            r["critic"],
            unsafe_allow_html=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
ResearchMind · Built with Streamlit + LangChain
</div>
""", unsafe_allow_html=True)
