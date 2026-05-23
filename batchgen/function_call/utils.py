"""Helpers shared by the function-call detectors (partial-JSON, schemas, etc.)."""

from __future__ import annotations

from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import orjson
import partial_json_parser
from partial_json_parser.core.options import Allow

from batchgen.function_call.core_types import Tool, ToolChoice


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


def _get_tool_schema_defs(tools: List[Tool]) -> dict:
    all_defs: Dict[str, Any] = {}
    for tool in tools:
        if tool.function.parameters is None:
            continue
        defs = tool.function.parameters.get("$defs", {})
        for def_name, def_schema in defs.items():
            if def_name in all_defs and all_defs[def_name] != def_schema:
                raise ValueError(
                    f"Tool definition '{def_name}' has multiple schemas, "
                    "which is not supported."
                )
            all_defs[def_name] = def_schema
    return all_defs


def _get_tool_schema(tool: Tool) -> dict:
    return {
        "properties": {
            "name": {"type": "string", "enum": [tool.function.name]},
            "parameters": (
                tool.function.parameters
                if tool.function.parameters
                else {"type": "object", "properties": {}}
            ),
        },
        "required": ["name", "parameters"],
    }


def infer_type_from_json_schema(schema: Dict[str, Any]) -> Optional[str]:
    """Infer the primary type ('string', 'number', 'object', 'array', ...) from a JSON Schema.

    Walks `type`/`anyOf`/`oneOf`/`allOf`/`enum`/`properties`/`items` in priority
    order. Returns None when the schema gives no hint.
    """
    if not isinstance(schema, dict):
        return None

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        if isinstance(type_value, list) and type_value:
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"

    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types: List[str] = []
        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(sub_schema)
                if inferred_type:
                    types.append(inferred_type)
            if types:
                if len(set(types)) == 1:
                    return types[0]
                if "string" in types:
                    return "string"
                return types[0]

    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")
        if len(enum_types) == 1:
            return enum_types.pop()
        return "string"

    if "allOf" in schema and isinstance(schema["allOf"], list):
        for sub_schema in schema["allOf"]:
            inferred_type = infer_type_from_json_schema(sub_schema)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    if "properties" in schema:
        return "object"

    if "items" in schema:
        return "array"

    return None


def get_json_schema_constraint(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["required"]],
    parallel_tool_calls: bool = True,
) -> Optional[dict]:
    """Build a JSON-array schema constraint for `tool_choice="required"` or a named tool."""
    if isinstance(tool_choice, ToolChoice):
        fn_name = tool_choice.function.name
        for tool in tools:
            if tool.function.name == fn_name:
                schema = {
                    "type": "array",
                    "minItems": 1,
                    "items": _get_tool_schema(tool),
                }
                if not parallel_tool_calls:
                    schema["maxItems"] = 1
                return schema
        return None

    if tool_choice == "required":
        json_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [_get_tool_schema(tool) for tool in tools],
            },
        }
        if not parallel_tool_calls:
            json_schema["maxItems"] = 1
        json_schema_defs = _get_tool_schema_defs(tools)
        if json_schema_defs:
            json_schema["$defs"] = json_schema_defs
        return json_schema

    return None
