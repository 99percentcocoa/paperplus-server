from app.services.communication import is_valid_image_message


def test_valid_image_message_extracts_url():
    message = {"from": "+911234567890", "content": {"type": "image", "image": {"link": "http://x/y.jpg"}}}
    is_valid, url = is_valid_image_message(message)
    assert is_valid is True
    assert url == "http://x/y.jpg"


def test_valid_image_message_tries_url_key_too():
    message = {"content": {"type": "image", "image": {"url": "http://x/y.jpg"}}}
    is_valid, url = is_valid_image_message(message)
    assert is_valid is True
    assert url == "http://x/y.jpg"


def test_filters_out_delivery_receipts():
    message = {"callback_type": "delivered", "content": {"type": "image", "image": {"link": "http://x/y.jpg"}}}
    is_valid, url = is_valid_image_message(message)
    assert is_valid is False
    assert url is None


def test_rejects_non_image_content():
    message = {"content": {"type": "text", "text": {"body": "hi"}}}
    is_valid, url = is_valid_image_message(message)
    assert is_valid is False
    assert url is None


def test_rejects_missing_content():
    assert is_valid_image_message({}) == (False, None)
