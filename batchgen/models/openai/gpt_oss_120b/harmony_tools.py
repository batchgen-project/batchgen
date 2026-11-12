# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""OpenAI function tools rendered as the harmony ``functions`` namespace.

GPT-OSS reads the functions it may call from the developer message of the
harmony prompt, written as TypeScript-like declarations::

    # Tools

    ## functions

    namespace functions {

    // Get the weather.
    type get_weather = (_: {
    // City name
    location: string,
    unit?: "c" | "f", // default: c
    }) => any;

    } // namespace functions

Each tool's JSON-schema ``parameters`` becomes the argument type. Inside the
supported subset below, the text is the harmony reference renderer's output
byte for byte, including its layout of nested objects (a described object's
comment repeats after the property name; nested braces close at the property
indent). The one deliberate difference: a multi-line description or title is
commented line by line, where the reference comments only the first line and
leaves the rest as bare text inside the type.

Supported subset, per schema node:

* ``type``: ``string``, ``number``, ``integer`` (rendered ``number``),
  ``boolean``, ``array``, ``object``; or a list of the scalar types and
  ``null`` (``string | null``);
* ``enum`` of strings on a ``string`` node (``"a" | "b"``);
* ``items`` (``T[]``; ``Array<any>`` without ``items``), ``properties``,
  ``required`` (optional properties get ``?``),
  ``additionalProperties: true|false`` (ignored);
* ``description``, ``title``, ``examples`` as comments; ``default`` as a
  trailing ``// default: ...`` comment;
* validation keywords that do not change the value type (``format``,
  ``minimum``, ``maxLength``, ``pattern``, ...) are ignored;
* a node without ``type`` and without type-bearing keywords is ``any``.

Anything else (``anyOf``/``oneOf``/``allOf``, ``$ref``/``$defs``, ``const``,
non-string ``enum``, tuple ``items``, map-typed ``additionalProperties``, a
type list naming ``object`` or ``array``, unknown keywords) is not converted
into a misleading type: the property it appears in is declared ``any`` and its
raw JSON schema is written in a ``// JSON schema: ...`` comment above it, so
the model still sees the full contract. When the top-level ``parameters``
itself is outside the subset, the whole argument is ``any`` with the schema
comment under the function's description.

Malformed tool definitions (not ``{"type": "function", "function": {...}}``,
missing or unusable name, duplicate names, non-object ``parameters``) raise
:class:`ValueError`; the scheduler rejects that request with the message.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, List, Mapping, Optional, Sequence

TOOLS_SECTION_HEADER = "# Tools\n\n## functions\n\nnamespace functions {\n\n"
TOOLS_SECTION_FOOTER = "\n\n} // namespace functions"

#: Prefix of every tool recipient (``to=functions.NAME``) and tool author.
FUNCTIONS_NAMESPACE = "functions"

_INDENT_STEP = "    "
_SCALAR_TYPES = {"string": "string", "number": "number", "integer": "number",
                 "boolean": "boolean"}
_TYPE_LIST_MEMBERS = frozenset(_SCALAR_TYPES) | {"null"}
_COMMENT_KEYS = frozenset({"description", "title", "examples", "default"})
_STRUCTURE_KEYS = frozenset(
    {"type", "enum", "items", "properties", "required", "additionalProperties"}
)
#: Validation keywords: they narrow values without changing the value type.
_IGNORED_KEYS = frozenset({
    "format", "pattern", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems",
    "maxItems", "uniqueItems", "minProperties", "maxProperties", "$schema",
    "$comment", "deprecated", "readOnly", "writeOnly",
})
_ALLOWED_KEYS = _COMMENT_KEYS | _STRUCTURE_KEYS | _IGNORED_KEYS
_FLOAT_EXPONENT_RE = re.compile(r"e([+-]?)0*(\d)")


def validate_function_name(name: Any, where: str) -> str:
    """A function name usable as the ``functions.NAME`` recipient.

    The harmony header ends a recipient at whitespace, and ``<|`` would open
    a special token, so names containing either cannot round-trip.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: function name must be a non-empty string, got {name!r}")
    if any(ch.isspace() for ch in name) or "<|" in name:
        raise ValueError(
            f"{where}: function name {name!r} contains whitespace or '<|', "
            f"which the harmony recipient header cannot carry"
        )
    return name


def render_functions_namespace(tools: Sequence[Any]) -> str:
    """The developer-message body declaring ``tools`` (OpenAI format)."""
    if not isinstance(tools, (list, tuple)):
        raise ValueError(f"tools must be a list of tool definitions, got {type(tools).__name__}")
    declarations: List[str] = []
    seen = set()
    for index, tool in enumerate(tools):
        where = f"tools[{index}]"
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise ValueError(
                f"{where}: only {{\"type\": \"function\", \"function\": {{...}}}} "
                f"tools are supported, got {_short_json(tool)}"
            )
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(f"{where}: 'function' must be an object, got {_short_json(function)}")
        name = validate_function_name(function.get("name"), where)
        if name in seen:
            raise ValueError(f"{where}: duplicate function name {name!r}")
        seen.add(name)
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"{where}: description must be a string, got {_short_json(description)}")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError(
                f"{where}: parameters must be a JSON-schema object, got {_short_json(parameters)}"
            )
        declarations.append(_render_function(name, description, parameters))
    return TOOLS_SECTION_HEADER + "\n\n".join(declarations) + TOOLS_SECTION_FOOTER


def _render_function(
    name: str, description: Optional[str], parameters: Optional[Mapping]
) -> str:
    out = _comment("", description) if description else ""
    if parameters is None:
        return f"{out}type {name} = () => any;"
    if _is_supported(parameters):
        argument = _type_expr(parameters, "")
    else:
        out += _schema_comment("", parameters)
        argument = "any"
    return f"{out}type {name} = (_: {argument}) => any;"


def _type_expr(schema: Mapping, indent: str) -> str:
    """TypeScript text of one supported schema node.

    ``indent`` is the indent of the node's own lines (object properties and
    closing brace); a property's value is rendered one step deeper than the
    property line, as the reference renderer does.
    """
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(_SCALAR_TYPES.get(k, k) for k in kind)
    if kind == "object":
        out = ""
        description = schema.get("description")
        if isinstance(description, str):
            out += _comment(indent, description)
        out += "{\n"
        required = set(schema.get("required") or ())
        for key, value in (schema.get("properties") or {}).items():
            out += _property(key, value, key in required, indent)
        return out + indent + "}"
    if kind == "string" and schema.get("enum"):
        return " | ".join(f'"{value}"' for value in schema["enum"])
    if kind in _SCALAR_TYPES:
        return _SCALAR_TYPES[kind]
    if kind == "array":
        items = schema.get("items")
        if items is None:
            return "Array<any>"
        return _type_expr(items, indent) + "[]"
    return "any"  # no type, no type-bearing keywords (checked by _is_supported)


def _property(key: str, schema: Any, required: bool, indent: str) -> str:
    out = ""
    fields = schema if isinstance(schema, Mapping) else {}
    title = fields.get("title")
    if isinstance(title, str):
        out += _comment(indent, title) + f"{indent}//\n"
    description = fields.get("description")
    if isinstance(description, str):
        out += _comment(indent, description)
    examples = fields.get("examples")
    if isinstance(examples, list) and examples:
        out += f"{indent}// Examples:\n"
        for example in examples:
            if isinstance(example, str):
                out += f'{indent}// - "{example}"\n'
    if _is_supported(schema):
        type_text = _type_expr(schema, indent + _INDENT_STEP)
    else:
        out += _schema_comment(indent, schema)
        type_text = "any"
    default = ""
    if "default" in fields:
        value = fields["default"]
        if isinstance(value, str):
            default = (
                f" // default: {value}" if "enum" in fields
                else f' // default: "{value}"'
            )
        else:
            default = f" // default: {_compact_json(value)}"
    optional = "" if required else "?"
    return f"{out}{indent}{key}{optional}: {type_text},{default}\n"


def _is_supported(schema: Any) -> bool:
    """True when ``schema`` lies inside the subset this module converts."""
    if not isinstance(schema, Mapping):
        return False
    keys = set(schema)
    if not keys <= _ALLOWED_KEYS:
        return False
    for text_key in ("description", "title"):
        if text_key in schema and not isinstance(schema[text_key], str):
            return False
    if "examples" in schema and not isinstance(schema["examples"], list):
        return False
    kind = schema.get("type")
    typed_keys = keys & {"enum", "items", "properties", "required", "additionalProperties"}
    if "type" not in schema:
        return not typed_keys
    if isinstance(kind, list):
        return (
            bool(kind)
            and all(isinstance(k, str) and k in _TYPE_LIST_MEMBERS for k in kind)
            and any(k != "null" for k in kind)
            and not typed_keys
        )
    if kind == "string":
        enum = schema.get("enum")
        return typed_keys <= {"enum"} and (
            enum is None
            or (isinstance(enum, list) and bool(enum) and all(isinstance(v, str) for v in enum))
        )
    if kind in ("number", "integer", "boolean"):
        return not typed_keys
    if kind == "array":
        if not typed_keys <= {"items"}:
            return False
        return "items" not in schema or _is_supported(schema["items"])
    if kind == "object":
        if not typed_keys <= {"properties", "required", "additionalProperties"}:
            return False
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return (
            isinstance(properties, Mapping)
            and all(isinstance(k, str) for k in properties)
            and isinstance(required, list)
            and all(isinstance(r, str) for r in required)
            and isinstance(schema.get("additionalProperties", True), bool)
        )
    return False


def _comment(indent: str, text: str) -> str:
    return "".join(f"{indent}// {line}\n" for line in text.split("\n"))


def _schema_comment(indent: str, schema: Any) -> str:
    return f"{indent}// JSON schema: {json.dumps(schema, ensure_ascii=False)}\n"


def _compact_json(value: Any) -> str:
    """JSON as the reference renderer prints a non-string default."""
    if isinstance(value, float) and math.isfinite(value):
        # Shortest round-trip digits, exponent written without '+' or
        # leading zeros (1e20, 1.5e-7).
        return _FLOAT_EXPONENT_RE.sub(r"e\1\2", repr(value)).replace("e+", "e")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _short_json(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."
