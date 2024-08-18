import numpy as np
from pydantic import BaseModel, field_validator, Field


class OcrTextOutout(BaseModel):
    bbox: list[float]
    text: str
    confidence: float

    @field_validator("bbox", mode="before")
    def convert_bbox_into_four(cls, v):
        if isinstance(v[0], list):
            v = combine_ocr_bbox(v)
        return v


class OcrFileOutput(BaseModel):
    filename: str
    ocr_result: list[OcrTextOutout]


class RoomInfo(BaseModel):
    name: str| None = None
    dimension: list[str]|str|None = None


def combine_ocr_bbox(bboxes):
    all_bboxes = np.array(bboxes)
    return [all_bboxes[:,0].min(), all_bboxes[:,1].min(), all_bboxes[:,2].max(), all_bboxes[:,3].max()]
