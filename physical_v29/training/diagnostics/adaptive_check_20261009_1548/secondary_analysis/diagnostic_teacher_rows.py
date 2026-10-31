"""Row selection for diagnostic component tests, never production inference."""
def combine(body, donor, rows):
    assert body.shape==donor.shape and len(body.shape)==2 and body.shape[1]==23
    assert len(rows)==len(set(rows)) and all(isinstance(j,int) and 0<=j<23 for j in rows)
    result=body.clone() if hasattr(body,'clone') else body.copy()
    result[:,rows]=donor[:,rows]
    return result
