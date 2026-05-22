"""
temporal-leaks
~~~~~~~~~~~~~~

Valgrind for Time-Series ML — automatically detect look-ahead bias in
data science pipelines using the Temporal Perturbation Test.

Quick start
-----------
>>> import pandas as pd
>>> from temporal_leaks import TemporalAudit
>>>
>>> auditor = TemporalAudit(mode="nullify", random_seed=42)
>>> report = auditor.check(df, timestamp_col="ts", pipeline_fn=my_features)
>>> print(report)

Decorator API
-------------
>>> from temporal_leaks import temporal_audit
>>>
>>> @temporal_audit(timestamp_col="ts")
... def build_features(df):
...     df = df.copy()
...     df["roll_mean"] = df["value"].rolling(3, min_periods=1).mean()
...     return df
"""

from temporal_leaks.auditor import AuditReport, PerturbationMode, TemporalAudit, temporal_audit
from temporal_leaks.exceptions import ColumnLeakMeta, TemporalLeakageError

__all__ = [
    "TemporalAudit",
    "AuditReport",
    "TemporalLeakageError",
    "ColumnLeakMeta",
    "PerturbationMode",
    "temporal_audit",
]

__version__ = "0.1.0"
__author__ = "temporal-leaks contributors"
