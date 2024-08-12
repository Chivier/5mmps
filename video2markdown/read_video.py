import base64
import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from ocrmac import ocrmac


# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def preprocess_image(image):
    gray = ImageOps.grayscale(image)
    # increase contrast
    contrast = ImageEnhance.Contrast(gray)
    adjusted = contrast.enhance(1.5)  # 对比度控制 (1.0-3.0)
    return adjusted


def extract_text_regions(image):
    """
    Extracts text regions from a given image.

    This function converts the input image to a NumPy array, finds contours in the image,
    and extracts bounding rectangles for contours that meet certain size criteria.

    Parameters:
    image (PIL.Image.Image): The input image from which text regions are to be extracted.

    Returns:
    list of tuple: A list of tuples, where each tuple contains the coordinates (x, y) and
                   dimensions (width, height) of a text region.
    """
    np_image = np.array(image)
    contours, _ = cv2.findContours(np_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions = []
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Check if the bounding rectangle meets certain size criteria
        if w > 50 and h > 10:
            text_regions.append((x, y, w, h))
    return text_regions


def judge_image_quality_mac(image_path):
    image = Image.open(image_path)

    preprocessed_image = preprocess_image(image)

    text_regions = extract_text_regions(preprocessed_image)

    total_confidence = 0
    num_chars = 0

    for (x, y, w, h) in text_regions:
        # crop region of text
        roi = image.crop((x, y, x + w, y + h))

        # extract text
        annotations = ocrmac.OCR(roi).recognize()

        for text, confidence, _ in annotations:
            num_chars += len(text)
            total_confidence += confidence

    avg_confidence = total_confidence / num_chars if num_chars > 0 else 0

    # Calculate readability score
    readability_score = (avg_confidence / 100) * num_chars

    # Normalize readability score to be between 0 and 100
    readability_score = min(max(readability_score, 0), 100)

    return readability_score


def blur_image(image, gaussianblur, save=""):
    # Apply Gaussian filter
    result_image = cv2.GaussianBlur(image, (gaussianblur, gaussianblur), 0)
    if save != "":
        cv2.imwrite(save, result_image)
    return result_image


def read_image(image_path):
    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return


def compare_info_similarity(info1, confidence1, info2, confidence2):
    """
    Compare the similarity between two pieces of information.
    :param info1:
    :param info2:
    :return:
    """
    pass


def extract_frames(video_path, output_folder, frame_interval=1):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame if it's the right interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()


class VideoItem:
    def __init__(self, video_path):
        self.path = video_path
        self.file = None
        self.frame_path = ''.join(video_path[:video_path.rfind('.')]) + '_frames'
        self.clear_frames = []
        print(self.frame_path)
        if not os.path.exists(self.frame_path):
            os.makedirs(self.frame_path)

    def read_video(self):
        extract_frames(self.path, self.frame_path, 10)

    def extract_clear_frames(self):
        self.clear_frames = []
        frame_score = []
        for file in os.listdir(self.frame_path):
            image_path = os.path.join(self.frame_path, file)
            score = judge_image_quality_mac(image_path)
            frame_score.append(score)
        for frame_id in range(1, len(frame_score) - 1):
            if frame_id in self.clear_frames:
                continue
            if frame_score[frame_id] >= frame_score[frame_id - 1] and frame_score[frame_id] >= frame_score[
                frame_id + 1]:
                if frame_score[frame_id] > 0.1:
                    # add windowsize = 2 into clear_frames
                    begin_id = max(0, frame_id - 2)
                    end_id = min(len(frame_score), frame_id + 3)
                    for i in range(begin_id, end_id):
                        if i not in self.clear_frames:
                            self.clear_frames.append(i)
            if frame_score[frame_id] > 0.1:
                if frame_id not in self.clear_frames:
                    self.clear_frames.append(frame_id)

    def merging_frames(self):
        pass
