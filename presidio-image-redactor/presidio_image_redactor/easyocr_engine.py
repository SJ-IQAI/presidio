import numpy as np
import easyocr

from presidio_image_redactor import OCR


class EasyOCREngine(OCR):
    """OCR class that uses EasyOCR for text detection."""

    def __init__(self, langs=None):
        if langs is None:
            langs = ["en"]
        self.reader = easyocr.Reader(langs)

    def perform_ocr(self, image, **kwargs):
        """Perform OCR on a given image.

        :param image: PIL Image/numpy array or file path(str) to be processed
        :param kwargs: Additional values for EasyOCR readtext

        :return: results dictionary containing bboxes and text for each detected word
        """
        img = np.array(image)
        results = self.reader.readtext(img, **kwargs)

        output = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
        }

        for bbox, text, conf in results:
            # Split multi-word detections into individual words
            # to match the per-word format expected by the analyzer
            words = text.split()
            if not words:
                continue

            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            x_max = int(max(p[0] for p in bbox))
            y_max = int(max(p[1] for p in bbox))
            total_width = x_max - x_min
            box_height = y_max - y_min

            if len(words) == 1:
                output["text"].append(text)
                output["left"].append(x_min)
                output["top"].append(y_min)
                output["width"].append(total_width)
                output["height"].append(box_height)
                output["conf"].append(conf * 100)
            else:
                # Approximate per-word bounding boxes by splitting
                # the total width proportionally by character count
                total_chars = sum(len(w) for w in words)
                current_x = x_min
                for word in words:
                    word_width = max(
                        1, int(total_width * len(word) / total_chars)
                    )
                    output["text"].append(word)
                    output["left"].append(current_x)
                    output["top"].append(y_min)
                    output["width"].append(word_width)
                    output["height"].append(box_height)
                    output["conf"].append(conf * 100)
                    current_x += word_width

        return output
