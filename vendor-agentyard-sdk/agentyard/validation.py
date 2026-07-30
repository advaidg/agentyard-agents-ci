"""Input/output schema validation using JSON Schema."""

from typing import Any


def validate_input(data: dict, schema: dict | None) -> tuple[bool, str | None]:
    """Validate input against declared input_schema. Returns (valid, error_message)."""
    if not schema:
        return True, None

    props = schema.get("properties", {})
    required = schema.get("required", [])

    for key in required:
        if key not in data:
            return False, f"Missing required field: {key}"

    for key, value in data.items():
        if key in props:
            expected_type = props[key].get("type")
            if expected_type and not _type_matches(value, expected_type):
                return False, (
                    f"Field '{key}' expected type '{expected_type}', "
                    f"got '{type(value).__name__}'"
                )

    return True, None


def validate_output(data: dict, schema: dict | None) -> tuple[bool, str | None]:
    """Validate output against declared output_schema."""
    return validate_input(data, schema)


def _type_matches(value: Any, json_type: str) -> bool:
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = type_map.get(json_type)
    if expected is None:
        return True
    return isinstance(value, expected)
