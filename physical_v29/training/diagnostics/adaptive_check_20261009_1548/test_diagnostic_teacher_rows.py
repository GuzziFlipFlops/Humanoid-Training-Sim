import json
from pathlib import Path
import numpy as np
from diagnostic_teacher_rows import combine
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(4631);a=rng.normal(size=(64,23));b=rng.normal(size=(64,23));checks=[]
for rows in [[0,1],[0,1,2,3,4,5,6,7],[],list(range(23))]:
    c=combine(a,b,rows);other=[i for i in range(23) if i not in rows]
    assert np.array_equal(c[:,rows],b[:,rows]) and np.array_equal(c[:,other],a[:,other])
    assert not np.shares_memory(c,a) and not np.shares_memory(c,b)
    checks.append(dict(rows=rows,passed=True))
for invalid in [[23],[-1],[1,1],[1.5]]:
    try:combine(a,b,invalid)
    except AssertionError:checks.append(dict(rejected=invalid,passed=True))
    else:raise AssertionError(invalid)
assert len(checks)==8
(OUT/'diagnostic_teacher_rows_CPU.json').write_text(json.dumps(dict(passed=True,checks=checks,scope='Row preservation only; not physical validation or production certification.'),indent=2)+'\n')
print('Eight row selection and rejection fixtures passed.')
