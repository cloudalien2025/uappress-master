import os
import streamlit as st

def ci_smoke_enabled() -> bool:
    return os.getenv("UAPPRESS_CI_SMOKE", "").strip() == "1" or os.getenv("CI", "").strip() == "1"

def mark_run_done():
    # Stable Playwright contract marker
    st.markdown("TEST_HOOK:RUN_DONE")
