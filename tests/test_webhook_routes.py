import unittest
from unittest.mock import patch

from flask import Flask

from routes.webhook_routes import webhook_bp


class WebhookLoggingTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(webhook_bp)
        self.client = self.app.test_client()

    @patch("routes.webhook_routes.setup_logging")
    @patch("threading.Thread")
    def test_ignores_status_callbacks_without_setting_up_logging(self, mock_thread, mock_setup_logging):
        payload = {
            "whatsapp": {
                "messages": [
                    {
                        "from": "919999999999",
                        "callback_type": "message_status",
                        "status": "sent"
                    }
                ]
            }
        }

        response = self.client.post("/webhook", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "ok")
        mock_setup_logging.assert_not_called()
        mock_thread.assert_not_called()

    @patch("routes.webhook_routes.setup_logging")
    @patch("threading.Thread")
    def test_logs_only_incoming_user_messages(self, mock_thread, mock_setup_logging):
        payload = {
            "whatsapp": {
                "messages": [
                    {
                        "from": "919999999999",
                        "callback_type": "incoming_message",
                        "content": {"type": "image", "image": {"url": "https://example.com/image.jpg"}}
                    }
                ]
            }
        }

        response = self.client.post("/webhook", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "ok")
        mock_setup_logging.assert_called_once()
        mock_thread.assert_called_once()


if __name__ == "__main__":
    unittest.main()
