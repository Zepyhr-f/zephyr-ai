from app.api.main import app


def test_new_capability_routes_are_registered() -> None:
    routes = set(app.openapi()["paths"])

    assert "/api/v1/knowledge/stats" in routes
    assert "/api/v1/rag/retrieve" in routes
    assert "/api/v1/rag/debug" in routes
    assert "/api/v1/conversations/{conversation_id}" in routes
    assert "/api/v1/feedback" in routes
    assert "/api/v1/admin/diagnostics" in routes
