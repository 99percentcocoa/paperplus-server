"""Shared fake VisionClient/CommunicationClient implementations, reused across
test_submission_service.py (calls handle_incoming_image directly) and test_webhook.py (drives
the same fakes through the real /webhook route via FastAPI dependency_overrides). No real
HTTP/Exotel calls happen in either case.
"""

from app.domain.errors import VisionClientError
from shared.contracts import ProcessingResult


class FakeVisionClient:
    def __init__(self, result: ProcessingResult):
        self._result = result

    def process(self, image_path, correlation_id, template_hint=None):
        return self._result


class FailingFakeVisionClient:
    def process(self, image_path, correlation_id, template_hint=None):
        raise VisionClientError("connection refused")


class FakeCommunicationClient:
    def __init__(self):
        self.sent_messages = []
        self.sent_images = []

    def send_message(self, to_number, message):
        self.sent_messages.append((to_number, message))

    def send_image(self, to_number, image_url, caption=""):
        self.sent_images.append((to_number, image_url, caption))
