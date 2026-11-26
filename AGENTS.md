# Language

Use UK English spelling in project prose and project-defined identifiers.
Preserve the documented spelling of external names and APIs.

# Design documentation

`design.org` is the source of truth for the system design.
Read it before planning or changing system behaviour.
When the design changes, update the relevant sections in the same change.
Replace superseded descriptions rather than appending amendments or explanations of what changed.
Keep only the current design, not a decision log.

Write prose in `design.org` with one sentence per source line.
Preserve paragraph breaks and Org headings, lists, tables and blocks.

# Implementation

Prefer maintained libraries for generic infrastructure when they reduce custom code and maintenance without adding unnecessary complexity.
Keep the core research logic in the project, and do not add dependencies merely to replace trivial helpers.
Use msgspec for typed configuration loading rather than hand-written field and type validation.

# External integrations

Before implementing or changing an external integration, read the current official API specifications rather than relying on memory.
Check authentication, endpoints, pagination, batching, rate limits, retry semantics and required headers.
Use the documented quota scope and response headers to govern request pacing and retries; do not assume different endpoints have independent allowances.
