from app.schemas import HealthOut
from app.auth import create_token, decode_token, hash_password, verify_password
from types import SimpleNamespace


def test_health_contract():
    result = HealthOut(api="healthy", database="healthy", model="loaded")
    assert result.model_dump() == {"api": "healthy", "database": "healthy", "model": "loaded"}


def test_password_hash_is_salted_and_verifiable():
    first = hash_password("correct horse battery staple")
    second = hash_password("correct horse battery staple")
    assert first != second
    assert verify_password("correct horse battery staple", first)
    assert not verify_password("wrong password", first)


def test_jwt_round_trip():
    token = create_token(SimpleNamespace(id=7, email="user@example.com"))
    payload = decode_token(token)
    assert payload["sub"] == "7"
    assert payload["email"] == "user@example.com"
