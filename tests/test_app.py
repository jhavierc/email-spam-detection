def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and b"SpamShield" in r.data


def test_generate_routes_into_two_folders(client):
    r = client.post("/api/generate", json={"count": 20, "spam_ratio": 0.5})
    assert r.status_code == 200
    delivered = r.get_json()["delivered"]
    assert len(delivered) == 20
    stats = client.get("/api/stats").get_json()
    assert stats["total"] == 20
    assert stats["folders"]["inbox"]["total"] + stats["folders"]["spam"]["total"] == 20
    for e in delivered:
        assert e["folder"] == ("spam" if e["prediction"]["label"] == "spam" else "inbox")
    inbox = client.get("/api/emails?folder=inbox").get_json()
    assert all(e["folder"] == "inbox" for e in inbox)


def test_compose_read_move_delete(client):
    r = client.post("/api/compose", json={"subject": "You WIN a free prize", "body": "Claim now"})
    email = r.get_json()["delivered"][0]
    assert email["folder"] == "spam" and email["true_label"] is None
    uid = email["uid"]
    assert client.get("/api/stats").get_json()["folders"]["spam"]["unread"] == 1
    assert client.get(f"/api/emails/{uid}").get_json()["read"] is True
    moved = client.patch(f"/api/emails/{uid}", json={"folder": "inbox"}).get_json()
    assert moved["folder"] == "inbox" and moved["user_moved"] is True
    assert client.delete(f"/api/emails/{uid}").status_code == 204
    assert client.get(f"/api/emails/{uid}").status_code == 404


def test_validation_errors(client):
    assert client.post("/api/compose", json={}).status_code == 400
    assert client.get("/api/emails?folder=trash").status_code == 400
    assert client.post("/api/generate", json={"count": "x"}).status_code == 400


def test_reset(client):
    client.post("/api/generate", json={"count": 3})
    assert client.post("/api/reset").get_json()["total"] == 0
