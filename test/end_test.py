from video2markdown.read_video import *

if __name__ == '__main__':
    # extract video frames, and calculate the score
    video_path = "../sample/video_test.mp4"
    video_item = VideoItem(video_path)
    video_item.describe_clear_frames()
