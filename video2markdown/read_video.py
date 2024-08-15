import json
import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from ocrmac import ocrmac
from skimage.metrics import structural_similarity as ssim

from video2markdown.utils import *


# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_frame_clear(frame, threshold=100.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold


def compare_images(image1, image2):
    # Convert the images to grayscale
    grayA = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM between the two images
    score, diff = ssim(grayA, grayB, full=True)
    return score


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


def extract_frames(video_path, output_folder, frame_interval=2):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # remove the output folder rm -rf output_folder
        os.system(f'rm -rf {output_folder}')
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
        if frame_count % frame_interval == 0 and is_frame_clear(frame):
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
        self.score_file = self.frame_path + '/score.json'
        self.description_file = self.frame_path + '/description.jsonl'
        self.clear_frames = []
        self.final_clear_frames = []
        self.final_clear_frames_score = []
        self.frame_score = []
        if not os.path.exists(self.frame_path):
            os.makedirs(self.frame_path)

    def read_video(self):
        extract_frames(self.path, self.frame_path, 10)

    def merge_similar_frames(self):
        self.final_clear_frames = [self.clear_frames[0]]
        self.final_clear_frames_score = [self.frame_score[self.clear_frames[0]]]
        index = 1
        while index < len(self.clear_frames):
            # compare current frame with last frame in final_clear_frames
            frame1 = cv2.imread(os.path.join(self.frame_path, f'frame_{self.final_clear_frames[-1]:04d}.jpg'))
            frame2 = cv2.imread(os.path.join(self.frame_path, f'frame_{self.clear_frames[index]:04d}.jpg'))
            if compare_images(frame1, frame2) > 0.70:
                clear_score1 = self.frame_score[self.final_clear_frames[-1]]
                clear_score2 = self.frame_score[self.clear_frames[index]]
                # keep the clear one
                if clear_score1 < clear_score2:
                    self.final_clear_frames[-1] = self.clear_frames[index]
                    self.final_clear_frames_score[-1] = clear_score2
            else:
                self.final_clear_frames.append(self.clear_frames[index])
                self.final_clear_frames_score.append(self.frame_score[self.clear_frames[index]])
            index += 1

        # rename the clear frames
        clear_id = 0
        for i in range(len(self.final_clear_frames)):
            frame_path = os.path.join(self.frame_path, f'frame_{self.final_clear_frames[i]:04d}.jpg')
            new_frame_path = os.path.join(self.frame_path, f'clear_frame_{clear_id:04d}.jpg')
            os.rename(frame_path, new_frame_path)
            clear_id += 1
        # remove unclear frames
        for file in os.listdir(self.frame_path):
            if file.startswith('frame_'):
                os.remove(os.path.join(self.frame_path, file))
        # save the score of clear frames
        score_info = {"score": self.final_clear_frames_score}
        with open(self.score_file, 'w') as f:
            json.dump(score_info, f)


    def extract_clear_frames(self):
        self.clear_frames = []
        self.frame_score = []
        for file in os.listdir(self.frame_path):
            if not file.endswith('.jpg'):
                continue
            image_path = os.path.join(self.frame_path, file)
            score = judge_image_quality_mac(image_path)
            self.frame_score.append(score)
        # go through the self.frame_score list
        for frameId in range(1, len(self.frame_score) - 1):
            # if the frame is already in clear_frames, skip
            if frameId in self.clear_frames:
                continue
            # if the frame is the peak of the score, add it into clear_frames
            if self.frame_score[frameId] >= self.frame_score[frameId - 1] and self.frame_score[frameId] >= \
                    self.frame_score[frameId + 1]:
                if self.frame_score[frameId] > 0.1:
                    # add windowsize = 2 into clear_frames
                    begin_id = max(0, frameId - 2)
                    end_id = min(len(self.frame_score), frameId + 3)
                    for i in range(begin_id, end_id):
                        if i not in self.clear_frames:
                            self.clear_frames.append(i)
            # if the frame is clear enough, add it into clear_frames
            if self.frame_score[frameId] > 0.1:
                if frameId not in self.clear_frames:
                    self.clear_frames.append(frameId)

    def describe_clear_frames(self):
        # read the score of clear frames
        with open(self.score_file, 'r') as f:
            score_info = json.load(f)
            self.final_clear_frames_score = score_info['score']
        description_raw = self.description_file
        # clear the description_raw file
        with open(description_raw, 'w') as f:
            pass
        for file in os.listdir(self.frame_path):
            if not file.startswith('clear_frame_'):
                continue
            image_path = os.path.join(self.frame_path, file)
            description = openai_read_image(image_path)
            info = {
                "filename": file,
                "score": self.final_clear_frames_score[int(file[12:16])],
                "description": description
            }
            with open(description_raw, 'a') as f:
                f.write(json.dumps(info) + '\n')
        # sort the description_raw file by filename
        with open(description_raw, 'r') as f:
            lines = f.readlines()
            lines.sort(key=lambda x: x[16:20])
        # save the sorted description file
        with open(self.description_file, 'w') as f:
            for line in lines:
                f.write(line)


