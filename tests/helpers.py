from encephalon.contracts import Header, Scope, Validity

SCOPE = Scope("demo-a", "0.1", "s-a")


def header(*, scope=SCOPE, at=0, start=None, end=None, revision=0, state=0,
           request_id=None, status="valid", quality=1.0, source_ids=("fixture-source",), provenance=(), reason=None):
    if end is None:
        end = at
    if start is None:
        start = end
    return Header(**scope.to_dict(), state_version=state, event_start_ms=start, event_end_ms=end,
                  available_at_ms=at, source_ids=source_ids, provenance=provenance,
                  validity=Validity(status, quality, at-end, reason), revision=revision, request_id=request_id)
