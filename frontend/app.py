import streamlit as st
import pandas as pd
from search import ForensicSearch

st.set_page_config(page_title="FORENIX AI", layout="wide")

st.title("🔍 FORENIX AI - CCTV Forensic Search System")

# Initialize
fs = ForensicSearch()

# Sidebar filters
st.sidebar.header("Filters")

object_query = st.sidebar.text_input("Object (person, car, bag)")
min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.5)
flagged_only = st.sidebar.checkbox("Flagged only")

# Buttons
search_btn = st.sidebar.button("Search")
summary_btn = st.sidebar.button("Show Summary")

# ---------------------------
# 🔍 SEARCH RESULTS
# ---------------------------
if search_btn:
    results = fs.query(
        object_class=object_query if object_query else None,
        min_confidence=min_conf,
        flagged_only=flagged_only
    )

    if results:
        st.subheader("📋 Search Results")

        df = pd.DataFrame([dict(r) for r in results])

        st.dataframe(df, use_container_width=True)

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            "results.csv",
            "text/csv"
        )

    else:
        st.warning("No results found")

# ---------------------------
# 📊 SUMMARY DASHBOARD
# ---------------------------
if summary_btn:
    stats = fs.get_summary()

    st.subheader("📊 Database Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Detections", stats["total"])
    col2.metric("Distinct Objects", stats["distinct_objects"])
    col3.metric("Flagged Events", stats["flagged"])

    st.write("### Object Distribution")

    df = pd.DataFrame(list(stats["by_class"].items()), columns=["Object", "Count"])
    st.bar_chart(df.set_index("Object"))

# Footer
st.markdown("---")
st.caption("FORENIX AI • CCTV Video Forensic Analysis System")