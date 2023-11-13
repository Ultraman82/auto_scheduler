from pydantic import BaseModel


class TeamsMessage(BaseModel):
    writer: str
    comment: str


class ReleaseItem(BaseModel):
    release_id: int
    item_description: str
    comment: str
    writer: str
    former_release_status: bool
    former_release_quantity: int
    updated_release_status: bool
    updated_release_quantity: int


class ReleaseItem(BaseModel):
    item_description: str
    comment: str
    writer: str
    release_id: int
    former_release_status: bool
    former_release_quantity: int
    updated_release_status: bool
    updated_release_quantity: int


class ReleaseItemPost(BaseModel):
    targetRail: str
    sourceRail: str
    qty: int
    reason: str


class Feedback(BaseModel):
    release_id: int
    writer: str
    comment: str
    item: str

class CartData(BaseModel):
    data: str

class ReleaseItem2(BaseModel):
    des: str
    dstock: int
    dtransit: int
    family: str
    gy3: int
    gyhaas: int
    hstock: int
    htransit: int
    in_transit: int
    is_3meter: bool
    is_blank: bool
    is_hass: bool
    kanban_w8: int
    length: int
    model: str
    processing: int
    processing_str: str
    rail_type: str
    release: str
    release_qty: int
    source_rail: str
    source_stock: int
    stock: int
    w9: int
