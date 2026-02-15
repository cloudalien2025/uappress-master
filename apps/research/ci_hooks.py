import os
import streamlit as st


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def ci_smoke_enabled() -> bool:
    return (
        _is_truthy(os.getenv("UAPPRESS_CI_SMOKE", "")) or
        _is_truthy(os.getenv("CI", ""))
    )


def mark_run_done():
    # Stable Playwright contract marker
    st.markdown("TEST_HOOK:RUN_DONE")
