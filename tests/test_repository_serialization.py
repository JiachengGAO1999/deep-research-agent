import json

from app.db.repository import _json_dump
from app.models.evidence import ResearchClaim


def test_json_dump_serializes_pydantic_models_inside_lists():
    rendered = _json_dump(
        [
            ResearchClaim(
                claim_text="A source-grounded claim.",
                validation_status="validated",
            )
        ]
    )
    decoded = json.loads(rendered)
    assert isinstance(decoded[0], dict)
    assert decoded[0]["claim_text"] == "A source-grounded claim."
