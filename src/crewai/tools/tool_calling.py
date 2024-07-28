from typing import Any, Dict, Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField
from pydantic.v1 import BaseModel, Field
from langchain_core.utils.pydantic import TBaseModel


class ToolCalling(PydanticBaseModel):
    tool_name: str = PydanticField(..., description="The name of the tool to be called.")
    arguments: Optional[Dict[str, Any]] = PydanticField(
        ..., description="A dictionary of arguments to be passed to the tool."
    )


class InstructorToolCalling(PydanticBaseModel):
    tool_name: str = PydanticField(
        ..., description="The name of the tool to be called."
    )
    arguments: Optional[Dict[str, Any]] = PydanticField(
        ..., description="A dictionary of arguments to be passed to the tool."
    )
