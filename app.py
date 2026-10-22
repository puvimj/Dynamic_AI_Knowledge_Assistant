"""
Dynamic AI Knowledge Assistant - Streamlit entry point.

    streamlit run app.py

Pages live in pages/ (thin files) and app_pages/ (the page logic). Theme: .streamlit/config.toml + app_pages/ui.py.
"""

import streamlit as st

from app_pages import ui
from app_pages.common import ASSETS, sidebar_workspace

st.set_page_config(page_title="Dynamic AI Knowledge Assistant", page_icon=":material/menu_book:", layout="wide")
st.logo(str(ASSETS / "logo.svg"), size="large")

home = st.Page("pages/0_Home.py", title="Home", icon=":material/home:", default=True)
builder = st.Page("pages/1_Dataset_Builder.py", title="Dataset Builder", icon=":material/library_add:")
library = st.Page("pages/2_Knowledge_Base.py", title="Knowledge Base", icon=":material/menu_book:")
chat = st.Page("pages/3_Chat_Assistant.py", title="Chat Assistant", icon=":material/forum:")
analytics = st.Page("pages/4_Analytics.py", title="Analytics", icon=":material/monitoring:")
evaluation = st.Page("pages/5_Evaluation.py", title="Evaluation", icon=":material/fact_check:")

page = st.navigation({"": [home], "Library": [builder, library], "Assistant": [chat],
                      "Insights": [analytics, evaluation]})

ui.inject_theme({"Home": "home", "Dataset Builder": "builder", "Knowledge Base": "kb", "Chat Assistant": "chat",
                 "Analytics": "analytics", "Evaluation": "eval"}.get(page.title, "home"))
sidebar_workspace()
page.run()
