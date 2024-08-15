import unittest
from unittest.mock import patch, MagicMock
from video2markdown.utils import openai_read_image
import os
from dotenv import load_dotenv


class TestUtils(unittest.TestCase):
    def test_read_image(self):
        load_dotenv()
        image_path = "../sample/video_test_frames/frame_0018.jpg"
        response = openai_read_image(image_path)
        print(response)
        self.assertGreater(len(response), 0)


if __name__ == '__main__':
    unittest.main()
