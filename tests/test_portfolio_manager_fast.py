from flask import Flask
import research.portfolio_manager_routes as pm


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_basic_portfolio_endpoint_returns_account_and_positions_and_caches(monkeypatch):
    pm._BASIC_CACHE.update({"at": 0, "payload": None})
    pm._CACHE.update({"at": 0, "payload": None})
    monkeypatch.setattr(pm.Config, "ALPACA_API_KEY", "test-key")
    monkeypatch.setattr(pm.Config, "ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setattr(pm.Config, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if url.endswith("/v2/account"):
            return FakeResponse({
                "equity": "1000", "cash": "400", "buying_power": "800",
                "portfolio_value": "1000",
            })
        if url.endswith("/v2/positions"):
            return FakeResponse([{
                "symbol": "NVDA", "qty": "2", "side": "long",
                "current_price": "300", "avg_entry_price": "250",
                "market_value": "600", "cost_basis": "500",
                "unrealized_pl": "100", "unrealized_plpc": "0.2",
            }])
        raise AssertionError("Unexpected Alpaca URL: " + url)

    monkeypatch.setattr(pm.requests, "get", fake_get)
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(pm.portfolio_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["logged_in"] = True

    response = client.get("/api/portfolio-manager?basic=1")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["account"]["equity"] == 1000
    assert payload["summary"]["position_count"] == 1
    assert payload["positions"][0]["symbol"] == "NVDA"
    assert payload["positions"][0]["market_value"] == 600
    assert len(calls) == 2

    cached = client.get("/api/portfolio-manager?basic=1")
    assert cached.status_code == 200
    assert cached.get_json()["positions"][0]["symbol"] == "NVDA"
    assert len(calls) == 2


def test_basic_portfolio_endpoint_fails_fast_without_credentials(monkeypatch):
    pm._BASIC_CACHE.update({"at": 0, "payload": None})
    monkeypatch.setattr(pm.Config, "ALPACA_API_KEY", "")
    monkeypatch.setattr(pm.Config, "ALPACA_SECRET_KEY", "")
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(pm.portfolio_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    response = client.get("/api/portfolio-manager?basic=1")
    assert response.status_code == 503
    assert "credentials are missing" in response.get_json()["error"].lower()
