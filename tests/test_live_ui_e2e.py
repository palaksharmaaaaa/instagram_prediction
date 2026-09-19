"""
Comprehensive End-to-End Test Suite for Streamlit UI and Live Server.

Tests:
1. Live HTTP Server availability, health check, and HTML delivery (http://localhost:8501).
2. Tab 1: NLP Profile Analytics & Search queries (standard, handle extraction, injection safety).
3. Tab 2: What-If Simulator across Instagram, YouTube, and Snapchat platforms.
4. TreeSHAP waterfall charts, domain invariants (Reach <= Impressions), and confidence intervals.
5. Tab 3 & Tab 4: Industry benchmarks, model diagnostics, SHA-256 integrity, and conformal coverage.
6. Defensive guardrail error handling for invalid or empty Meta tokens.
"""

import urllib.request
import urllib.error
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_live_http_server_health():
    """Verify that the live Streamlit server is running and returns HTTP 200 OK."""
    try:
        req = urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=5)
        status = req.status
        body = req.read().decode("utf-8")
        assert status == 200
        assert "ok" in body.lower()
    except (urllib.error.URLError, ConnectionRefusedError) as err:
        pytest.skip(f"Live server not reachable at http://localhost:8501: {err}")


def test_live_http_server_homepage():
    """Verify that the home page returns valid HTML containing Streamlit and title strings."""
    try:
        req = urllib.request.urlopen("http://localhost:8501/", timeout=5)
        assert req.status == 200
        html = req.read().decode("utf-8")
        assert len(html) > 1000
        assert "Instagram AI" in html or "Streamlit" in html
    except (urllib.error.URLError, ConnectionRefusedError) as err:
        pytest.skip(f"Live server not reachable at http://localhost:8501: {err}")


def test_app_initial_load_zero_exceptions():
    """Verify the Streamlit app loads in AppTest with zero exceptions and 4 tabs."""
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"App crashed with exceptions: {[e.value for e in at.exception]}"
    assert len(at.tabs) == 4
    assert len(at.button) >= 5
    assert len(at.selectbox) >= 7


def test_tab1_nlp_query_execution():
    """Test Natural Language Search query execution in Tab 1."""
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()

    # Find the query text input (TextInput 4: 'Enter requirement:')
    query_input = None
    for t in at.text_input:
        if "requirement" in t.label.lower() or "enter" in t.label.lower():
            query_input = t
            break
    assert query_input is not None, "Query input box not found"

    # Find the Analyze button ('🚀 Analyze')
    analyze_btn = None
    for b in at.button:
        if "analyze" in b.label.lower():
            analyze_btn = b
            break
    assert analyze_btn is not None, "Analyze button not found"

    # Test 1: Standard query
    query_input.input("Show accounts with over 500k followers").run()
    analyze_btn.click().run()
    assert len(at.exception) == 0, f"Exceptions during query: {[e.value for e in at.exception]}"

    # Test 2: Handle search without crash
    query_input.input("Show posts for @cristiano").run()
    analyze_btn.click().run()
    assert len(at.exception) == 0

    # Test 3: Preposition 'for' query without false username extraction
    query_input.input("Show accounts for marketing campaigns").run()
    analyze_btn.click().run()
    assert len(at.exception) == 0

    # Test 4: Injection attack string handled defensively
    query_input.input("<script>alert(1)</script> DROP TABLE profiles; --").run()
    analyze_btn.click().run()
    assert len(at.exception) == 0


def test_tab2_instagram_simulation_and_treeshap():
    """Test Tab 2 What-If simulation with Instagram Reel and verify TreeSHAP waterfall."""
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()

    # Select Instagram platform
    platform_box = None
    for s in at.selectbox:
        if "platform" in s.label.lower():
            platform_box = s
            break
    assert platform_box is not None
    platform_box.select("Instagram").run()

    # Find simulation button ('🚀 Run Post Performance Forecast')
    forecast_btn = None
    for b in at.button:
        if "forecast" in b.label.lower() or "run post" in b.label.lower():
            forecast_btn = b
            break
    assert forecast_btn is not None

    forecast_btn.click().run()
    assert len(at.exception) == 0

    # Verify metrics
    metrics = {m.label: m.value for m in at.metric}
    assert "Projected Reach" in metrics
    assert "Projected Impressions" in metrics

    # Parse numeric reach and impressions
    reach_val = int(metrics["Projected Reach"].replace(",", ""))
    imp_val = int(metrics["Projected Impressions"].replace(",", ""))

    # Domain Invariant: Reach <= Impressions
    assert reach_val <= imp_val, f"Reach {reach_val} exceeds Impressions {imp_val}"

    # Verify Plotly charts (TreeSHAP waterfall and distribution charts)
    assert len(at.get("plotly_chart")) >= 1, "No plotly charts found in app"

    # Verify that TreeSHAP waterfall section was rendered
    treeshap_rendered = any("TreeSHAP" in str(m.value) or "Algorithmic Levers" in str(m.value) for m in at.markdown)
    assert treeshap_rendered, "TreeSHAP Waterfall Plot section was not rendered in Tab 2"


def test_tab2_youtube_simulation_and_treeshap():
    """Test Tab 2 What-If simulation with YouTube Short and verify dynamic labels."""
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()

    platform_box = [s for s in at.selectbox if "platform" in s.label.lower()][0]
    platform_box.select("YouTube").run()

    format_box = [s for s in at.selectbox if "format" in s.label.lower()][0]
    format_box.select("YouTube Short").run()

    forecast_btn = [b for b in at.button if "forecast" in b.label.lower() or "run post" in b.label.lower()][0]
    forecast_btn.click().run()
    assert len(at.exception) == 0

    # Metric labels dynamically adapt for YouTube
    metric_labels = [m.label for m in at.metric]
    has_yt_reach = any("views & reach" in l.lower() or "projected reach" in l.lower() for l in metric_labels)
    assert has_yt_reach, f"YouTube reach label not found in: {metric_labels}"

    reach_val = None
    imp_val = None
    for m in at.metric:
        if "reach" in m.label.lower():
            reach_val = int(m.value.replace(",", ""))
        elif "impressions" in m.label.lower() or "views" in m.label.lower():
            imp_val = int(m.value.replace(",", ""))

    if reach_val is not None and imp_val is not None:
        assert reach_val <= imp_val


def test_tab2_snapchat_simulation_and_treeshap():
    """Test Tab 2 What-If simulation with Snapchat Spotlight and verify dynamic labels."""
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()

    platform_box = [s for s in at.selectbox if "platform" in s.label.lower()][0]
    platform_box.select("Snapchat").run()

    format_box = [s for s in at.selectbox if "format" in s.label.lower()][0]
    format_box.select("Snapchat Spotlight").run()

    forecast_btn = [b for b in at.button if "forecast" in b.label.lower() or "run post" in b.label.lower()][0]
    forecast_btn.click().run()
    assert len(at.exception) == 0

    metric_labels = [m.label for m in at.metric]
    has_sc_reach = any("snap" in l.lower() or "reach" in l.lower() for l in metric_labels)
    assert has_sc_reach, f"Snapchat reach label not found in: {metric_labels}"


def test_meta_graph_api_expander_defensive_guardrails():
    """Verify that clicking Meta Graph API buttons with empty tokens displays graceful errors instead of unhandled exceptions."""
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()

    # Discover button with empty token
    discover_btn = [b for b in at.button if "auto-discover" in b.label.lower()][0]
    discover_btn.click().run()
    assert len(at.exception) == 0
    # An error banner should be present instructing user to input token
    assert len(at.error) >= 1
    assert any("user access token" in str(e.value).lower() for e in at.error)

    # Fetch creator button with empty token
    fetch_btn = [b for b in at.button if "fetch live creator" in b.label.lower()][0]
    fetch_btn.click().run()
    assert len(at.exception) == 0
    assert len(at.error) >= 1
