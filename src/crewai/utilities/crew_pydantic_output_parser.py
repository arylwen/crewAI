import json
from typing import Any, List, Type

import regex
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation
from langchain_core.pydantic_v1 import ValidationError
from pydantic import BaseModel 
# https://stackoverflow.com/questions/78099646/pydantic-error-subclass-of-basemodel-expected-type-type-error-subclass-expect
#from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.pydantic import (
    PYDANTIC_MAJOR_VERSION,
    PydanticBaseModel,
    TBaseModel,
)


class CrewPydanticOutputParser(PydanticOutputParser):
    """Parses the text into pydantic models"""

    pydantic_object: Type[TBaseModel]

    def parse_result(self, result: List[Generation]) -> Any:
        result[0].text = self._transform_in_valid_json(result[0].text)

        # Treating edge case of function calling llm returning the name instead of tool_name
        json_object = json.loads(result[0].text)
        if "tool_name" not in json_object:
            json_object["tool_name"] = json_object.get("name", "")
        result[0].text = json.dumps(json_object)

        # we probably need a separate class, PlannerPydanticOutputParser
        # convert to string if the planner returns a list at all; most open source
        # llm will return a structured object within the list of plans
        if "list_of_plans_per_task" in json_object:
            task_list = json_object['list_of_plans_per_task']
            if type(task_list) is list: 
                json_object['list_of_plans_per_task'] = [str(x) for x in task_list]

        try:
            print(f"crew_pydantic_output_parser: about to validate: {str(json_object).encode(errors='ignore').decode(errors='ignore')}")
            return self.pydantic_object.model_validate(json_object)
        except ValidationError as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
            raise OutputParserException(msg, llm_output=json_object)
        except Exception as e:
            print(f"crew_pydantinc_output_parser: {e}")
            raise e

    def _transform_in_valid_json(self, text) -> str:
        text = text.replace("```", "").replace("json", "")
        json_pattern = r"\{(?:[^{}]|(?R))*\}"
        matches = regex.finditer(json_pattern, text)

        for match in matches:
            try:
                # Attempt to parse the matched string as JSON
                json_obj = json.loads(match.group())
                # Return the first successfully parsed JSON object
                json_obj = json.dumps(json_obj)
                return str(json_obj)
            except json.JSONDecodeError:
                # If parsing fails, skip to the next match
                continue
        return text
