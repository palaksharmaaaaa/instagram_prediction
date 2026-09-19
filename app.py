import streamlit as st
import sys

sys.path.append("src")

from pipeline import run_pipeline


st.set_page_config(
    page_title="Instagram AI Analytics",
    layout="wide"
)


st.title("Instagram Analytics & Prediction System")

st.write(
    "Ask a question about Instagram accounts "
    "using natural language."
)


prompt = st.text_input(
    "Enter your requirement",
    placeholder=(
        "Example: Give me accounts above "
        "5 million followers"
    )
)


if st.button("Run"):

    if not prompt.strip():

        st.warning(
            "Please enter a requirement."
        )

    else:

        with st.spinner("Processing..."):

            requirements, result = run_pipeline(
                prompt
            )

        # -------------------------
        # Requirements
        # -------------------------

        st.subheader(
            "Detected Requirements"
        )

        st.json(requirements)

        # -------------------------
        # Result
        # -------------------------

        st.subheader(
            f"Results: {len(result)} accounts"
        )

        if result.empty:

            st.warning(
                "No accounts matched your requirement."
            )

        else:

            display_columns = [
                "Username",
                "Followers",
                "Country",
                "Main topic",
                "Main video category",
                "Likes Avg.",
                "Comments Avg.",
                "Views Avg.",
                "Engagement Rate"
            ]

            if "Predicted Reach" in result.columns:

                display_columns.append(
                    "Predicted Reach"
                )

            if "Predicted Impressions" in result.columns:

                display_columns.append(
                    "Predicted Impressions"
                )

            display_columns = [
                col
                for col in display_columns
                if col in result.columns
            ]

            st.dataframe(
                result[display_columns],
                use_container_width=True
            )