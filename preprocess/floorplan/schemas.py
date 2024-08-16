from pydantic import BaseModel, field_validator, Field

from datasets.schemas.document import Entity, LxdmDocument, LxdmSplitDocument
from datasets.utils import combine_bbox


class OcrTextOutout(BaseModel):
    bbox: list[float]
    text: str
    confidence: float

    @field_validator("bbox", mode="before")
    def convert_bbox_into_four(cls, v):
        if isinstance(v[0], list):
            v = combine_bbox(v)
        return v


class OcrFileOutput(BaseModel):
    filename: str
    ocr_result: list[OcrTextOutout]


class RoomInfo(BaseModel):
    name: str| None = None
    dimension: list[str]|str|None = None


class FloorplanEntity(Entity):
    total_area: float| None = Field(None, alias="total area")
    rooms: list[RoomInfo]| None = None


class FloorplanDocument(LxdmDocument):
    entity: FloorplanEntity


class FloorplanSplitDocument(LxdmSplitDocument):
    entity: FloorplanEntity
