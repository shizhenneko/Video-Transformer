
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline import VideoPipeline, ProcessResult
from exceptions import KeyExhaustedError

class TestKeyRotation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "system": {"output_dir": "output"},
            "analyzer": {},
            "proxy": {},
            "validator": {},
        }
        self.logger = MagicMock()
        self.api_counter = MagicMock()
        self.api_counter.current_count = 0
        self.pipeline = VideoPipeline(self.config, self.logger, self.api_counter)

    @patch("pipeline.ContentAnalyzer")
    @patch("pipeline.VideoDownloader")
    @patch("pipeline.VideoPipeline._allocate_gemini_key")
    def test_key_rotation_logic(self, mock_allocate_key, mock_downloader, mock_analyzer_cls):
        # Setup mocks
        mock_allocate_key.side_effect = ["key_1", "key_2", "key_3"]
        
        # Mock Downloader to return a dummy path
        mock_downloader_instance = mock_downloader.return_value
        mock_downloader_instance.download_video.return_value = Path("dummy_video.mp4")

        # Mock Analyzer instance
        mock_analyzer_instance = mock_analyzer_cls.return_value
        
        # Scenario:
        # 1. First key ("key_1") fails with KeyExhaustedError
        # 2. Second key ("key_2") succeeds
        
        # We need to mock analyze_video to raise KeyExhaustedError first, then return a result
        mock_result = MagicMock()
        mock_result.knowledge_doc.deep_dive = []
        mock_result.knowledge_doc.visual_schema = ""
        mock_result.knowledge_doc.to_markdown.return_value = ""

        mock_analyzer_instance.analyze_video.side_effect = [
            KeyExhaustedError("Simulated exhaustion"),
            mock_result
        ]
        
        # Run pipeline
        result = self.pipeline.process_single_video("http://example.com/video")
        
        # Assertions
        self.assertTrue(result.success)
        self.assertEqual(mock_allocate_key.call_count, 2)
        self.assertEqual(mock_allocate_key.call_args_list[0][0], ()) # First call
        # Check logs or print to verify rotation happened
        # (In a real test we'd check logger calls, but here we just want to ensure it didn't crash and retried)
        
        print(f"Result success: {result.success}")
        print(f"Allocate key called {mock_allocate_key.call_count} times")

    @patch("pipeline.ContentAnalyzer")
    @patch("pipeline.VideoDownloader")
    @patch("pipeline.VideoPipeline._allocate_gemini_key")
    def test_max_rotation_exhaustion(self, mock_allocate_key, mock_downloader, mock_analyzer_cls):
        # Scenario: All keys fail
        mock_allocate_key.return_value = "key_x"
        mock_downloader_instance = mock_downloader.return_value
        mock_downloader_instance.download_video.return_value = Path("dummy_video.mp4")
        
        mock_analyzer_instance = mock_analyzer_cls.return_value
        mock_analyzer_instance.analyze_video.side_effect = KeyExhaustedError("Simulated exhaustion")
        
        result = self.pipeline.process_single_video("http://example.com/video")
        
        self.assertFalse(result.success)
        self.assertIn("Key pool exhausted", result.error_message)
        # Should attempt 10 times (max_key_rotations)
        self.assertEqual(mock_analyzer_instance.analyze_video.call_count, 10)
        print(f"Result success: {result.success}")
        print(f"Analyze video called {mock_analyzer_instance.analyze_video.call_count} times")

if __name__ == "__main__":
    unittest.main()
