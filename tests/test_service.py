"""
Unit and integration tests for ThinkAI Studio services.
Follows security best practices:
- Uses mocked API clients for regular unit tests (zero unexpected network/paid API calls).
- Never contains real or hardcoded API keys or service account credentials.
- Gracefully skips live integration tests unless explicitly enabled via RUN_LIVE_API_TESTS=true.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# Load local environment (if .env exists)
load_dotenv()

from services.gemini_service import (
    GeminiService,
    get_gemini_client,
    mask_api_key,
    sanitize_error_message,
    RECOMMENDED_MODELS,
)


class TestGeminiServiceUnit(unittest.TestCase):
    """Hermetic unit tests using mocked responses and synthetic dummy tokens."""

    def test_mask_api_key(self):
        """Verify API key masking never leaks full keys."""
        self.assertEqual(mask_api_key(None), "Not Set")
        self.assertEqual(mask_api_key(""), "Not Set")
        self.assertEqual(mask_api_key("short"), "***")

        # Use an obvious synthetic test string
        dummy_test_key = "MOCK_AI_STUDIO_API_KEY_1234567890_XYZ"
        masked = mask_api_key(dummy_test_key)
        self.assertTrue(masked.startswith("MOCK_A"))
        self.assertTrue(masked.endswith("_XYZ"))
        self.assertIn("...", masked)
        self.assertNotIn("1234567890", masked)

    def test_sanitize_error_message_redacts_key(self):
        """Verify sensitive strings are scrubbed from errors."""
        dummy_secret = "TESTING_SECRET_KEY_9876543210"
        raw_error = Exception(f"Failed to connect using key={dummy_secret} due to internal error.")
        sanitized = sanitize_error_message(raw_error, api_key=dummy_secret)
        self.assertNotIn(dummy_secret, sanitized)
        self.assertIn("[REDACTED", sanitized)

    def test_sanitize_error_message_known_codes(self):
        """Verify standard HTTP/API errors return user-friendly advice."""
        err_404 = Exception("404 NOT_FOUND. Model not found.")
        self.assertIn("Model unavailable or deprecated", sanitize_error_message(err_404))

        err_503 = Exception("503 UNAVAILABLE. High demand spikes.")
        self.assertIn("high demand", sanitize_error_message(err_503).lower())

        err_429 = Exception("429 RESOURCE_EXHAUSTED. Quota exceeded.")
        self.assertIn("quota", sanitize_error_message(err_429).lower())

        err_auth = Exception("403 API_KEY_INVALID. Forbidden.")
        self.assertIn("Authentication Error", sanitize_error_message(err_auth))

    def test_client_init_missing_credentials(self):
        """Verify missing credentials raise clear, actionable ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                get_gemini_client(api_key="")
            self.assertIn("Authentication credentials missing", str(ctx.exception))

    def test_client_init_missing_service_account_file(self):
        """Verify non-existent GOOGLE_APPLICATION_CREDENTIALS path raises clear error."""
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/path/sa.json", "GEMINI_API_KEY": ""}):
            with self.assertRaises(ValueError) as ctx:
                get_gemini_client(api_key=None)
            self.assertIn("GOOGLE_APPLICATION_CREDENTIALS file not found", str(ctx.exception))

    @patch("services.gemini_service.genai.Client")
    def test_client_init_with_api_key(self, mock_client_cls):
        """Verify client initializes properly with an API key."""
        dummy_key = "MOCK_KEY_FOR_TEST_ONLY_12345"
        client = get_gemini_client(api_key=dummy_key)
        mock_client_cls.assert_called_once_with(api_key=dummy_key)
        self.assertIsNotNone(client)

    @patch("services.gemini_service.genai.Client")
    def test_generate_direct_mocked(self, mock_client_cls):
        """Verify generate_direct returns text using mocked model response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Mocked AI response for testing."
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = GeminiService(api_key="MOCK_KEY")
        result = service.generate_direct(
            model="gemini-3.5-flash-lite",
            prompt="Hello world",
        )
        self.assertEqual(result, "Mocked AI response for testing.")

    @patch("services.gemini_service.genai.Client")
    def test_stream_conversation_mocked(self, mock_client_cls):
        """Verify stream_conversation yields text tokens from mocked stream."""
        mock_client = MagicMock()
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "there!"
        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]
        mock_client_cls.return_value = mock_client

        service = GeminiService(api_key="MOCK_KEY")
        messages = [{"role": "user", "content": "Hi"}]
        chunks = list(service.stream_conversation(
            model="gemini-3.5-flash-lite",
            messages=messages,
        ))
        self.assertEqual(chunks, ["Hello ", "there!"])

    @patch("services.gemini_service.genai.Client")
    def test_verify_connection_mocked(self, mock_client_cls):
        """Verify connection health check with mocked success and failure."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Pong"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = GeminiService(api_key="MOCK_KEY")
        is_ok, msg = service.verify_connection()
        self.assertTrue(is_ok)
        self.assertIn("Connected", msg)

        # Test failure scenario
        mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE")
        is_ok, msg = service.verify_connection()
        self.assertFalse(is_ok)
        self.assertIn("high demand", msg.lower())

    def test_recommended_models_structure(self):
        """Verify recommended models list is well-formed."""
        self.assertGreater(len(RECOMMENDED_MODELS), 0)
        has_default = any(m.get("default") for m in RECOMMENDED_MODELS)
        self.assertTrue(has_default)
        for m in RECOMMENDED_MODELS:
            self.assertIn("id", m)
            self.assertIn("name", m)


class TestLiveGeminiIntegration(unittest.TestCase):
    """
    Live API integration tests.
    Disabled by default to protect against unexpected paid API calls or missing credentials in CI.
    Enable explicitly by setting RUN_LIVE_API_TESTS=true in your local environment.
    """

    def setUp(self):
        self.run_live = os.getenv("RUN_LIVE_API_TESTS", "").lower() in ("true", "1", "yes")
        self.api_key = os.getenv("GEMINI_API_KEY")

    def test_live_ping_and_generation(self):
        if not self.run_live:
            self.skipTest("Live integration tests disabled by default. Set RUN_LIVE_API_TESTS=true to enable.")

        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured. Skipping live test.")

        service = GeminiService(api_key=self.api_key)
        is_connected, msg = service.verify_connection(model="gemini-3.5-flash-lite")
        self.assertTrue(is_connected, f"Live connection verification failed: {msg}")


if __name__ == "__main__":
    unittest.main()
