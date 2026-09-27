"""Shared measured-quantity reduction; unknown is never zero."""
def measure(rows: list[dict], field: str, *, applicable=lambda row: True) -> dict:
    selected = [row for row in rows if applicable(row)]
    known = [row[field] for row in selected if row[field] is not None]
    return dict(known_subtotal=sum(known), actual_total=sum(known) if len(known) == len(selected) else None,
                known_count=len(known), unknown_count=len(selected)-len(known),
                not_applicable_count=len(rows)-len(selected), applicable_count=len(selected))

