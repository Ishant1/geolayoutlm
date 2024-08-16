import cv2
from skimage import io
import numpy as np
import pandas as pd
import pytesseract
from paddleocr import PaddleOCR

from datasets.floorplan.schemas import OcrTextOutout
from datasets.floorplan.utils import combine_ocr_bbox, normalise_bbox, join_block_and_words

OCR_ENGINE = r"C:\Users\iagg1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


class OcrEngine:
    def __init__(self, local_image=True) -> None:
        self.engine = None
        self.terradata_file:str = None

        if local_image:
            self.loader = self._load_local_image
        else:
            self.loader = self._load_web_image

        self.setup_ocr()

    def setup_ocr(self) -> None:
        self.engine = PaddleOCR(use_angle_cls=True, lang='en', det_lang='ml')
        self.engine_block = PaddleOCR(use_angle_cls=True, lang='en', det_lang=None)
        # pytesseract.pytesseract.tesseract_cmd = OCR_ENGINE

    @staticmethod
    def preprocess_image(img: np.ndarray) -> np.ndarray:
        if len(img.shape)==3 and img.shape[2]==3:
            img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype='uint8')
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        return img

    @staticmethod
    def clean_image(img):
        # Rescale the image, if needed.
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        # Converting to gray scale
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
        img = cv2.merge(result_planes)

        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)  # increases the white region in the image
        img = cv2.erode(img, kernel, iterations=1)  # erodes away the boundaries of foreground object

        # Apply blur to smooth out the edges
        # img = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply threshold to get image with only b&w (binarization)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Save the filtered image in the output directory
        # Recognize text with tesseract for python
        return img

    @staticmethod
    def _load_local_image(img_path: str) -> np.ndarray:
        gray_image = (io.imread(img_path, as_gray=True)*255).astype("uint8")
        if len(gray_image.shape)==3:
            if gray_image.shape[0]==1:
                return gray_image[0]
            else:
                raise ValueError(f"Wrong format of file {img_path}")
        return gray_image

    @staticmethod
    def _load_web_image(image_path: str) -> np.ndarray:
        return io.imread(image_path)

    def process_image_with_paddle(self, img: str, block=False, normalise=False) -> list[OcrTextOutout]:
        bbox_and_text = self.engine_block.ocr(img) if block else self.engine.ocr(img)
        # bbox_and_text = pytesseract.image_to_data(img,output_type='dict')
        all_outputs = []
        for ocr_output in bbox_and_text[0]:
            ocr_text_output = OcrTextOutout(
                bbox = ocr_output[0],
                text = ocr_output[1][0],
                confidence = ocr_output[1][1]
            )
            all_outputs.append(ocr_text_output)
        if normalise:
            s = io.imread(img).shape
            img_height, img_width = s[0], s[1]
            for o in all_outputs:
                o.bbox = normalise_bbox(o.bbox, img_height, img_width)
        return all_outputs

    def process_with_tesseract(self, img: np.ndarray):
        bbox_and_text = pytesseract.image_to_data(img, output_type='dict')
        return bbox_and_text

    def get_result_from_a_file(self, image_path: str, block=False, normalise=True):
        # img = self.loader(image_path)
        # processed_img = self.preprocess_image(img)
        padd_words = self.process_image_with_paddle(image_path, normalise=normalise)
        if block:
            padd_blocks = self.process_image_with_paddle(image_path, block=True, normalise=normalise)
            padd_df = pd.DataFrame(join_block_and_words(padd_blocks, padd_words))
            return padd_df

        return padd_words

    def get_result_with_tesseract(self, image_path: str):
        img = self.loader(image_path)
        # img = self.preprocess_image(img)
        tess_df = pd.DataFrame(self.process_with_tesseract(img))
        tess_df = tess_df.loc[
                  (tess_df['text'].str.strip() != '') , :
                  ]
        tess_df["bbox"] = tess_df.apply(lambda x: [x['left'], x['top'], x['left']+x['width'], x['top']+x['height']], axis=1)
        group_results = (tess_df.groupby(['block_num', 'line_num'])[['text', 'bbox']]
                         .apply(lambda x: [{'text':y['text'], 'bbox':y['bbox']} for i,y in x.iterrows()])).reset_index().rename({0 : 'dict'},axis=1)
        group_results['text'] = group_results['dict'].apply(lambda x: ' '.join([w['text'] for w in x]))
        group_results['bbox'] = group_results['dict'].apply(lambda x: combine_ocr_bbox([w['bbox'] for w in x]))

        return group_results


