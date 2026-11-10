"""Live, read-only Playwright smoke test for the deployed OptiTrade dashboard.

Set PLAYWRIGHT_DASHBOARD_PASSWORD in the GitHub Actions repository secrets to
enable authenticated Backtest / AI Crash checks. No trades or replay jobs are
started by this test.
"""
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

BASE_URL = os.getenv(
    "PLAYWRIGHT_TEST_BASE_URL",
    "https://tv-gpt-alp-v2-gpt3.onrender.com",
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
        page.on("pageerror", lambda error: page_errors.append(str(error) + " | " + str(getattr(error, "stack", ""))))
        page.on("console", lambda msg: page_errors.append(msg.text + " | location=" + str(msg.location)) if msg.type == "error" else None)

        # Render's free service may sleep or briefly return a gateway 503 while
        # waking. Retry transient gateway responses, but never accept a persistent 5xx.
        response = None
        transient_errors = []
        for attempt in range(4):
            try:
                response = page.goto(
                    BASE_URL + "/",
                    wait_until="domcontentloaded",
                    timeout=180_000,
                )
                if response is not None and response.status < 500:
                    break
                transient_errors.append(
                    f"attempt {attempt + 1}: HTTP {response.status if response else 'no response'}"
                )
            except Exception as exc:
                transient_errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(20)
        assert response is not None, (
            "The deployed app returned no navigation response after retries: "
            + " | ".join(transient_errors)
        )
        assert response.status < 500, (
            f"Dashboard remained unavailable after retries: {transient_errors}"
        )
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
        try:
            page.locator("#main-tabs").wait_for(state="visible", timeout=30_000)
        except PlaywrightTimeoutError:
            login_error = page.locator(".error")
            error_text = login_error.inner_text().strip() if login_error.count() else ""
            if "wrong password" in error_text.lower():
                page.screenshot(path=str(ARTIFACTS / "login-rejected.png"), full_page=True)
                browser.close()
                pytest.skip(
                    "The login page is reachable, but GitHub Actions secret DASHBOARD_PASSWORD "
                    "does not match Render's DASHBOARD_PASSWORD. Update the secret to enable "
                    "authenticated browser smoke tests."
                )
            raise AssertionError(
                "Login did not reveal the dashboard. Current URL: " + page.url
                + "; visible login error: " + error_text
            )
        assert page.locator("#pane-bt").count() == 1, "Backtest pane is missing."

        # Confirm the new Analytics account-history view is present and navigable.
        analytics_tab = page.locator("#main-tabs .tab").filter(has_text="Analytics")
        assert analytics_tab.count() == 1, "The Analytics navigation tab is missing."
        analytics_tab.click()
        page.wait_for_timeout(1000)
        parse_errors = page.evaluate("""() => Array.from(document.scripts).map((script, index) => {
          if (script.src || !script.textContent.trim()) return null;
          try { new Function(script.textContent); return null; }
          catch (error) { return {index, message: error.message, stack: error.stack}; }
        }).filter(Boolean)""")
        print("Inline JavaScript parse errors:", parse_errors)
        if parse_errors:
            inline_scripts = page.locator("script").all_text_contents()
            for parse_error in parse_errors:
                script_lines = inline_scripts[parse_error["index"]].splitlines()
                print("Live script source around parser line:", script_lines[3992:4001])
            script_inventory = page.evaluate("""() => Array.from(document.scripts).map((script, index) => ({index, src: script.src, length: script.textContent.length, preview: script.textContent.slice(0, 100)}))""")
            print("Browser script inventory:", script_inventory)
            cdp = context.new_cdp_session(page)
            cdp.send("Runtime.enable")
            for script_index, script_text in enumerate(inline_scripts):
                if not script_text.strip():
                    continue
                try:
                    compiled = cdp.send("Runtime.compileScript", {
                        "expression": script_text, "sourceURL": f"dashboard-inline-{script_index}.js",
                        "persistScript": False
                    })
                    if compiled.get("exceptionDetails"):
                        print("CDP JavaScript parse location:", compiled["exceptionDetails"])
                except Exception as compile_error:
                    print("CDP compile error:", str(compile_error))
            cdp.detach()
        assert not parse_errors, "Inline JavaScript syntax errors: " + str(parse_errors)
        assert not page_errors, "Browser JavaScript error after opening Analytics: " + " | ".join(page_errors)
        page.locator("#pane-ana8").wait_for(state="visible")
        page.locator("#avbtn-history").wait_for(state="attached")
        page.locator("#avbtn-history").click()
        page.locator("#av-history").wait_for(state="visible")
        assert page.locator("#ah-tbody").count() == 1, "Account transaction ledger table is missing."

        # Open Backtest and keep the AI Crash replay manual; do not trigger heavy replay.
        # Target the actual Backtest tab by its handler instead of fuzzy text matching.
        # Assert both the active-tab state and pane visibility so a click/UI regression
        # produces a specific diagnostic before we test the crash charts.
        backtest_tab = page.locator('#main-tabs .tab[onclick*="switchTab(12,\'bt\')"]')
        assert backtest_tab.count() == 1, "The Backtest navigation tab is missing or its handler changed."
        backtest_tab.click()
        page.wait_for_function(
            "() => document.querySelector('#main-tabs .tab[onclick*=\\\"switchTab(12,\\\\\\'bt\\\\\\')\\\"]')?.classList.contains('on') && document.querySelector('#pane-bt')?.classList.contains('on')",
            timeout=15_000,
        )
        page.locator("#pane-bt").wait_for(state="visible")
        crash_tab = page.locator("#bt-subnav .bt-stab", has_text="AI Crash")
        crash_tab.click()
        page.locator("#ai-crash-terminal").wait_for(state="visible")
        # The replay button may intentionally be hidden by a nested panel state;\n        # confirm it exists without requiring visibility or triggering the replay.\n        page.locator("#ai-crash-prod-btn").wait_for(state="attached")

        # Verify the real authenticated API payload as well as the rendered page.
        # Render may still be deploying the commit that triggered this workflow.
        # Retry while the deployed replay is stale, but never treat a stale HTTP 200 as success.
        api_result = None
        for attempt in range(10):
            api_result = page.evaluate("""async () => {
              const response = await fetch('/api/ai-crash/dashboard', {
                credentials: 'same-origin'
              });
              let body = {};
              try { body = await response.json(); } catch (_) {}
              return {status: response.status, body};
            }""")
            body = api_result.get("body") or {}
            replay_check = body.get("historical_replay") or {}
            try:
                end_date = datetime.strptime(replay_check.get("coverage_end", ""), "%Y-%m-%d").date()
                fresh_enough = end_date >= datetime.now(timezone.utc).date() - timedelta(days=10)
            except (TypeError, ValueError):
                fresh_enough = False
            if api_result.get("status") == 200 and fresh_enough:
                break
            if attempt < 9:
                time.sleep(15)
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
        replay = payload.get("historical_replay") or {}
        assert replay.get("coverage_start") and replay["coverage_start"] <= "2000-06-01", (
            "Historical replay does not cover the expected 2000 crash era: "
            + str(replay.get("coverage_start"))
        )
        assert replay.get("coverage_end") and replay["coverage_end"] >= (
            datetime.now(timezone.utc).date() - timedelta(days=10)
        ).isoformat(), (
            "Historical replay tail is stale: coverage ends " + str(replay.get("coverage_end"))
        )
        assert replay.get("available") is True, (
            "Historical crash replay is unavailable: " + str(replay.get("warning", "no warning returned"))
        )
        episodes = replay.get("episodes") or []
        episode_text = " ".join(str(e.get("episode", "")) for e in episodes).lower()
        expected_episodes = ("dot-com", "global financial crisis", "covid", "2022")
        missing_episodes = [name for name in expected_episodes if name not in episode_text]
        assert not missing_episodes, (
            "Historical replay missing expected crash episodes " + str(missing_episodes)
            + "; received " + str([e.get("episode") for e in episodes])
        )
        invalid = [
            e.get("episode") for e in episodes
            if e.get("status") == "Insufficient history"
            or not e.get("20pct_date")
            or e.get("score_at_peak") is None
            or e.get("max_score_pre_breach") is None
        ]
        assert not invalid, (
            "Historical crash validation is not complete for: " + str(invalid)
            + ". Coverage=" + str(replay.get("coverage_start")) + " to "
            + str(replay.get("coverage_end"))
        )
        print("Historical replay coverage:", replay.get("coverage_start"), "to", replay.get("coverage_end"))
        print("Historical replay sources:", replay.get("data_sources"))
        print("Historical episode results:", [
            {k: e.get(k) for k in ("episode", "peak_date", "20pct_date", "score_at_peak",
                                    "max_score_pre_breach", "first_60_date", "lead_days", "status")}
            for e in episodes
        ])

        page.wait_for_function(
            """() => {
              const body = document.querySelector('#ai-crash-body');
              const score = document.querySelector('#ai-crash-score');
              return body && body.style.display !== 'none' && score &&
                     score.textContent.trim() !== '—';
            }""",
            timeout=120_000,
        )
        # Assert the cross-asset data is present and recent before trusting blank/visible canvases.
        chart_data = payload.get("cross_asset_charts") or {}
        chart_coverage = {}
        for symbol in ("spy", "qqq"):
            series = chart_data.get(symbol) or {}
            points = series.get("points") or []
            latest_point = points[-1].get("t") if points else None
            chart_coverage[symbol] = {"source": series.get("source"), "points": len(points), "latest": latest_point}
            assert len(points) >= 100, f"{symbol.upper()} cross-asset history is too short: {chart_coverage[symbol]}"
            # Old bundled baselines are allowed for historical plotting only when
            # the API explicitly labels their freshness; the dashboard must warn users.
            age_days = series.get("age_days")
            assert isinstance(age_days, int) and age_days >= 0, (
                f"{symbol.upper()} cross-asset history is missing freshness metadata: {chart_coverage[symbol]}"
            )
            assert bool(series.get("is_stale")) == (age_days > 10), (
                f"{symbol.upper()} stale-data flag disagrees with its age: {chart_coverage[symbol]}"
            )
            if age_days > 10:
                print(f"WARNING: {symbol.upper()} chart data is {age_days} days old; dashboard must show stale-data warning.")
        print("Cross-asset chart coverage:", chart_coverage)

        # Check Chart.js actually instantiated the SPY, QQQ and rolling-correlation plots.
        rendered_charts = page.evaluate("""() => {
          const ids = ['ai-ca-spy', 'ai-ca-qqq', 'ai-ca-correlation'];
          return ids.map(id => {
            const canvas = document.getElementById(id);
            const chart = canvas && window.Chart ? Chart.getChart(canvas) : null;
            return {id, exists: !!canvas, display: canvas ? getComputedStyle(canvas).display : null,
                    chart: !!chart, labels: chart ? chart.data.labels.length : 0,
                    datasets: chart ? chart.data.datasets.map(d => ({label:d.label, points:d.data.length,
                      valid:d.data.filter(v => v !== null && Number.isFinite(Number(v))).length})) : []};
          });
        }""")
        print("Rendered cross-asset charts:", rendered_charts)
        for chart in rendered_charts:
            assert chart["exists"] and chart["chart"] and chart["labels"] >= 20, (
                "Expected a populated rendered chart, but got " + str(chart)
            )

        # Confirm short-window controls do not discard the return warm-up needed for correlation.
        for window in ("1m", "3m", "1y"):
            page.locator(f'#ai-ca-range-controls [data-ca-range="{window}"]').click()
            page.wait_for_function("""(id) => {
              const canvas = document.getElementById(id);
              const chart = canvas && window.Chart ? Chart.getChart(canvas) : null;
              return !!chart && chart.data.labels.length >= 20;
            }""", arg="ai-ca-correlation", timeout=10_000)

        page.screenshot(path=str(ARTIFACTS / "ai-crash-mobile.png"), full_page=True)

        # Capture a desktop-width rendering too, useful for table/layout regressions.
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.screenshot(path=str(ARTIFACTS / "ai-crash-desktop.png"), full_page=True)

        assert not page_errors, "Uncaught browser JavaScript errors: " + " | ".join(page_errors)
        context.close()
        browser.close()
