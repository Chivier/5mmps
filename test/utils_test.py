import unittest
from unittest.mock import patch, MagicMock
from video2markdown.utils import fetch_openai_envvars, openai_read_image
import os
from dotenv import load_dotenv


class TestUtils(unittest.TestCase):
    def fetch_openai_envvars_returns_api_key(self):
        load_dotenv()

        print(os.getenv("OPENAI_API_KEY"))
        print()
        print(fetch_openai_envvars())
        self.assertGreater(fetch_openai_envvars(), 0)

    # def openai_read_image_returns_response(self):
    #     base64_image = "test_base64_image"
    #     mock_response = MagicMock()
    #     mock_response.chat.completions.create.return_value = {"choices": [{"message": {"content": "Mocked response"}}]}
    #     with patch('openai.OpenAI', return_value=mock_response):
    #         response = openai_read_image(base64_image)
    #         self.assertEqual(response, {"choices": [{"message": {"content": "Mocked response"}}]})
    #
    # def openai_read_image_handles_empty_image(self):
    #     base64_image = ""
    #     mock_response = MagicMock()
    #     mock_response.chat.completions.create.return_value = {"choices": [{"message": {"content": "Mocked response"}}]}
    #     with patch('openai.OpenAI', return_value=mock_response):
    #         response = openai_read_image(base64_image)
    #         self.assertEqual(response, {"choices": [{"message": {"content": "Mocked response"}}]})


if __name__ == '__main__':
    unittest.main()
