"""
Unit tests for Vercel Serverless API endpoints (api/index.py).
Follows security best practices:
- Tests routes using Flask test client
- Hermetic tests with zero network/paid calls
- Verifies secrets are never leaked in API outputs
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from api.index import app


class TestVercelApi(unittest.TestCase):
    """Test suite for Vercel API and Web UI."""

    def setUp(self):
        self.client = app.test_client()

    def test_index_route_serves_html(self):
        """Verify the root route returns the ChatGPT-style SPA HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ThinkAI Studio", response.data)
        self.assertIn(b"How can I help you today?", response.data)

    def test_health_endpoint(self):
        """Verify the health endpoint reports status without leaking full key."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDummySecretKey123456789"}):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["has_api_key"])
            self.assertNotIn("AIzaSyDummySecretKey123456789", data["key_masked"])
            self.assertIn("...", data["key_masked"])

    def test_models_endpoint(self):
        """Verify the models endpoint returns recommended models."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("models", data)
        self.assertGreater(len(data["models"]), 0)

    def test_chat_endpoint_empty_message(self):
        """Verify 400 error when message is blank."""
        response = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    @patch("api.index.GeminiService")
    def test_chat_endpoint_success(self, mock_service_class):
        """Verify successful chat completion with mocked Gemini service."""
        mock_instance = MagicMock()
        mock_instance.stream_conversation.return_value = ["Hello", " from", " Gemini!"]
        mock_service_class.return_value = mock_instance

        payload = {
            "message": "Hi there",
            "model": "gemini-3.5-flash-lite",
            "temperature": 0.5,
            "history": [],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["reply"], "Hello from Gemini!")
        self.assertEqual(data["model"], "gemini-3.5-flash-lite")

    @patch("requests.post")
    def test_chat_endpoint_with_service_binding(self, mock_requests_post):
        """Verify chat forwards to internal service URL when THINKAI_STUDIO_URL is bound."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "reply": "Forwarded reply from internal thinkai-studio service",
            "model": "gemini-3.5-flash-lite",
        }
        mock_requests_post.return_value = mock_resp

        with patch.dict(os.environ, {"THINKAI_STUDIO_URL": "http://thinkai-studio.internal"}):
            payload = {"message": "Hello service binding", "model": "gemini-3.5-flash-lite"}
            response = self.client.post("/api/chat", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["reply"], "Forwarded reply from internal thinkai-studio service")
            mock_requests_post.assert_called_once()
            args, kwargs = mock_requests_post.call_args
            self.assertEqual(args[0], "http://thinkai-studio.internal/api/chat")

    @patch("requests.get")
    def test_health_endpoint_with_service_binding(self, mock_requests_get):
        """Verify health check queries bound internal service when THINKAI_STUDIO_URL is set."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "service": "ThinkAI Studio Internal",
            "has_api_key": True,
            "key_masked": "AQ.Ab8...45lQ",
        }
        mock_requests_get.return_value = mock_resp

        with patch.dict(os.environ, {"THINKAI_STUDIO_URL": "http://thinkai-studio.internal"}):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["bound_service"], "thinkai-studio")
            mock_requests_get.assert_called_once_with("http://thinkai-studio.internal/api/health", timeout=5)


if __name__ == "__main__":
    unittest.main()


