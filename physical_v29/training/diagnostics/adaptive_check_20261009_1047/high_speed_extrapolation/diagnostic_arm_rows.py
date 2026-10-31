"""Diagnostic action-row composition; truthful closed-loop feedback only."""
def replace_arm_rows(current, teacher, rows, mask):
    assert current.shape==teacher.shape and current.ndim==2
    assert rows.shape==(current.shape[1],) and mask.shape==(current.shape[0],)
    assert rows.dtype==mask.dtype==__import__('torch').bool
    result=current.clone()
    result[mask[:,None].expand_as(result) & rows[None,:].expand_as(result)] = teacher[mask[:,None].expand_as(teacher) & rows[None,:].expand_as(teacher)]
    return result
