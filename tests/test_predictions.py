from app.services.prediction_service import sanitize_filename


def test_sanitize_filename_removes_path_and_unsafe_characters():
    assert sanitize_filename(r"..\folder\satellite<script>.png") == "satellitescript.png"


def test_sanitize_filename_handles_missing_name():
    assert sanitize_filename(None) == "upload"
