---
schema: v1
prefix: XCTRACE
---

## XCTRACE-ARTIFACT-FIRST-CAPTURE-001
WHEN the xctrace recorder returns an error after writing a capture, the metalcounters SHALL accept the capture only after five validations succeed: workload marker, TOC export, required counter schemas, command-buffer selection, and report analysis.
