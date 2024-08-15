import unittest
import cv2
from video2markdown.read_video import *
import video2markdown


class ImageTest(unittest.TestCase):
    def test_image_score(self):
        # read image
        score = judge_image_quality_mac("../sample/image_test_1.png")
        print(score)
        score = judge_image_quality_mac("../sample/image_test_blur_1.jpg")
        print(score)
        score = judge_image_quality_mac("../sample/image_test_blur_2.jpg")
        print(score)
        score = judge_image_quality_mac("../sample/image_test_blur_3.jpg")
        print(score)
        self.assertGreater(score / 100, 0)

    def test_video_score(self):
        # extract video frames, and calculate the score
        video_path = "../sample/video_test.mp4"
        video_item = VideoItem(video_path)
        video_item.read_video()
        # read video scores
        files = os.listdir(video_item.frame_path)
        files = sorted(files)
        video_item.extract_clear_frames()
        video_item.merge_similar_frames()
        print(video_item.clear_frames)
        print(video_item.final_clear_frames)
        self.assertGreater(len(video_item.clear_frames), 0)

    def test_video_describe(self):
        # extract video frames, and calculate the score
        video_path = "../sample/video_test.mp4"
        video_item = VideoItem(video_path)
        video_item.describe_clear_frames()
        self.assertGreater(len(video_item.clear_frames), 0)


if __name__ == '__main__':
    unittest.main()
