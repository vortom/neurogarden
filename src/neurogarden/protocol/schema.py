"""The JSON Schema artifact derived from the message models: drift-tested, for other clients."""

from __future__ import annotations

import json

from .messages import PROTOCOL_VERSION, client_adapter, server_adapter

REF_TEMPLATE = "#/$defs/{model}"
SCHEMA_ID = (
    "https://raw.githubusercontent.com/vortom/neurogarden/main/protocol/v1/neurogarden.schema.json"
)


def export_schema() -> dict:
    client = client_adapter.json_schema(ref_template=REF_TEMPLATE)
    server = server_adapter.json_schema(ref_template=REF_TEMPLATE)
    defs = {**client.pop("$defs", {}), **server.pop("$defs", {})}
    defs["ClientMessage"] = client
    defs["ServerMessage"] = server
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": f"NeuroGarden protocol v{PROTOCOL_VERSION}",
        "description": "Envelope {v, type, payload}; unknown types are ignored by receivers.",
        "oneOf": [{"$ref": "#/$defs/ClientMessage"}, {"$ref": "#/$defs/ServerMessage"}],
        "$defs": dict(sorted(defs.items())),
    }


def schema_text() -> str:
    """Canonical text of the artifact: sorted keys, two-space indent, trailing newline."""
    return json.dumps(export_schema(), indent=2, sort_keys=True) + "\n"
