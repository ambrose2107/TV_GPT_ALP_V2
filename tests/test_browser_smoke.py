"""Live, read-only Playwright smoke test for the deployed OptiTrade dashboard.

Set PLAYWRIGHT_DASHBOARD_PASSWORD in the GitHub Actions repository secrets to
enable authenticated Backtest / AI Crash checks. No trades or replay jobs are
started by this test.
"""
import os
import re
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

BASE_URL = os.getenv(
    "PLAYWRIGHT_TEST_BASE_URL",
    "https://tv-gpt-alp-v2-gpt.onrender.com",
).rstrip("/")
PASSWORD = os.getenv("PLAYWRIGHT_DASHBOARD_PASSWORD", "")
ARTIFACTS = Path("test-results")


def test_deployed_dashboard_and_ai_crash_smoke():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=1,
            is_mobile=True,
            has_touch=True,
        )
        page = context.new_page()
        page.set_default_timeout(15_000)
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        # Render's free service may sleep. Give the cold start time to finish.
        response = page.goto(
            BASE_URL + "/",
            wait_until="domcontentloaded",
            timeout=180_000,
        )
        assert response is not None, "The deployed app returned no navigation response."
        assert response.status < 500, f"Dashboard returned HTTP {response.status}."
        page.screenshot(path=str(ARTIFACTS / "login-mobile.png"), full_page=True)
        assert page.locator('input[name="password"]').count() == 1, (
            "Expected the dashboard login screen; check the deployed URL or routing."
        )

        if not PASSWORD:
            browser.close()
            pytest.skip(
                "Set GitHub Actions secret DASHBOARD_PASSWORD to enable authenticated "
                "Backtest and AI Crash checks. Login-page smoke test passed."
            )

        page.locator('input[name="password"]').fill(PASSWORD)
        page.get_by_role("button", name=re.compile("login", re.I)).click()
        page.wait_for_url(re.compile(r".*/(?!login(?:\?.*)?$).*"), timeout=30_000)
        page.locator("#main-tabs").wait_for(state="visible")
        assert page.locator("#pane-bt").count() == 1, "Backtest pane is missing."

        # Open Backtest and keep the AI Crash replay manual; do not trigger heavy replay.
        page.locator("#main-tabs .tab", has_text="Backtest").click()
        page.locator("#pane-bt").wait_for(state="visible")
        crash_tab = page.locator("#bt-subnav .bt-stab", has_text="AI Crash")
        crash_tab.click()
        page.locator("#ai-crash-terminal").wait_for(state="visible")
        page.locator("#ai-crash-prod-btn").wait_for(state="visible")

        # Verify the real authenticated API payload as well as the rendered page.
        api_result = page.evaluate("""async () => {
          const response = await fetch('/api/ai-crash/dashboard', {
            credentials: 'same-origin'
          });
          let body = {};
          try { body = await response.json(); } catch (_) {}
          return {status: response.status, body};
        }""")
        assert api_result["status"] == 200, (
            "AI Crash dashboard API failed: "
            + str(api_result["status"])
            + " "
            + str(api_result["body"].get("error", ""))
        )
        payload = api_result["body"]
        assert "score" in payload, "AI Crash payload is missing its risk score."
        assert "us_market" in payload, "AI Crash payload is missing US-market risk."
        assert "reliability" in payload, "AI Crash payload is missing reliability metrics."

        page.wait_for_function(
            """() => {
              const body = document.querySelector('#ai-crash-body');
              const score = document.querySelector('#ai-crash-score');
              return body && body.style.display !== 'none' && score &&
                     score.textContent.trim() !== '—';
            }""",
            timeout=120_000,
        )
        page.screenshot(path=str(ARTIFACTS / "ai-crash-mobile.png"), full_page=True)

        # Capture a desktop-width rendering too, useful for table/layout regressions.
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.screenshot(path=str(ARTIFACTS / "ai-crash-desktop.png"), full_page=True)

        assert not page_errors, "Uncaught browser JavaScript errors: " + " | ".join(page_errors)
        context.close()
        browser.close()
