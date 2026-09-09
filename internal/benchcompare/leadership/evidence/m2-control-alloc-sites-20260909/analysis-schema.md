# Allocation-site per-invocation analysis contract v1

Prospective supplement to `protocol.md` and `schema.md`. No live profile or
matrix observation has been taken. This specifies the **per-invocation** offline
analyzer only; the matrix manifest, runner and paired summary need their own
review before execution. A successful invocation analysis does not validate a
matrix, decode pprof, establish attribution, or qualify GELU V2.

## Implementation and interface

New self-contained standard-library Ruby files in this evidence directory:
`analyze_sites.rb` and `analyze_sites_test.rb`, compatible with local Ruby 2.6.10.
Use a namespace `AllocationSites`, not globals shared with older analyzers.
Reuse the duplicate-key rejecting `Hash#[]=` pattern from the prior fixed-count
analyzer, but do not load that script: it has an unconditional CLI and different
schemas. Neither the earlier analyzer nor its evidence may be modified.

Public entry points:

- `AllocationSites.parse(text)` parses strict JSON without duplicate keys at any
  depth, NaN/Infinity, trailing content, or additions. It returns the raw object;
  schema-specific validation occurs in `analyze`.
- `AllocationSites.analyze(capture, tail)` validates all inputs and returns the
  derived object below, or raises `AllocationSites::Invalid` with a useful path
  and reason. It never executes a child, reads source, or changes raw objects.
- Guard CLI with `if $PROGRAM_NAME == __FILE__`. Invocation:
  `ruby analyze_sites.rb CAPTURE_JSON TAIL_JSON DERIVED_JSON`.
  Inputs and output must be absolute paths; the output must not exist. Read and
  analyze both inputs before exclusive creation of the output (0600). Serialize
  one JSON object and a newline. Errors print to stderr and exit nonzero. Input
  invalidity must not create a derived file. I/O failures may leave a partial
  output, which must not be removed or overwritten. No silent fallback or retry.

Do all arithmetic as Ruby integers with **explicit range checks**, not by relying
on arbitrary precision to accept values the protocol disallows. JSON float
tokens are rejected for integer fields even when mathematically integral.

## Input validation

Every object has exactly the keys in `schema.md`; reject missing, additional and
duplicate fields. Reject wrong primitive types, including null arrays, string or
floating counters, and booleans in place of integers. Validate all literal schema,
control, N, warmup, completed count, capacity, flags and environment values.
Require `region_error == ""`, `errors == []`, rate before/after 1, GMP in [1,12],
`goos == "darwin"`, `goarch == "arm64"`. Require a nonempty `go_version` string;
the future manifest validator must match it to the exact frozen build version.
Require the exact expected fully qualified caller and worker names in schema.md.

Counters are unsigned 64-bit (NumGC is widened but may be no greater than
2^32-1). All raw profile counters are signed 64-bit and must also be nonnegative
for valid analysis. Array ordinals/counts are nonnegative integers, never floats.
Snapshots must have ok=true, reported_count<=65536, that many rows, consecutive
ordinals, and exactly 32 canonical lowercase unpadded hex PC strings per row.
Each PC and frame PC fits uint64; `0x0` is allowed; reject nonzero PCs after the
first zero. Every frame has exactly pc/function/file/line, strings for names and
files, and a nonnegative integer line. An empty PC prefix requires frames=[];
an active row requires a nonempty PC prefix, nonempty frames, and every emitted
function name nonempty. Zero-active rows may retain unresolved names.

For each active row, require alloc_objects>0, positive alloc_bytes, exact
alloc_bytes/alloc_objects, and free fields no greater than allocated fields.
Derive a positive signed-64-bit slot size; check multiplication and require
free_bytes == free_objects * slot_size. alloc_objects=0 is valid only when all
four counters are zero; retain this row as size-unknown, without a size key.
Do not require 32 PCs to yield 32 frames: inline expansion changes cardinality.
Never sort or deduplicate an individual PC or frame stack.

Check cumulative total_alloc/mallocs/frees/num_gc are nondecreasing at every
ordered boundary, including tail. heap_alloc and heap_objects may decrease.
All boundary differences must fit signed 64-bit. Require the pre snapshot's
total_alloc and mallocs differences to be zero. Tail report_write_excluded must
be true. Do not reject a run merely because immediate totals and profile sums
differ; expose that discrepancy below.

## Aggregation and arithmetic

Define a profile-counter object with exactly alloc_bytes, free_bytes,
alloc_objects, free_objects. Every aggregate, sum, delta, product and discrepancy
must fit signed 64-bit. Check intermediates as well as final results.

Within each snapshot, key active rows by `[slot_size, nonzero_PC_prefix]`.
Sum all four counters, retain every contributor ordinal and its multiplicity.
Contributors of the same raw key must have identical ordered function-name
stacks, both within and across the two snapshots; differing location fields are
not a difference in identity. Reject inconsistent symbolization rather than
silently choosing a contributor. Compute post-minus-pre over the complete union;
missing sides contribute numeric zero and an empty ordinal list. Reject a
decrease in any cumulative profile field, even if another key increases enough
to hide it. Retain unchanged keys too.

Next key the raw-key union by `[slot_size, ordered_function_name_stack]`. Sum
pre/post/delta counters across every raw-key contributor, retaining all raw-key
indices and cardinality. This second collision layer must never overwrite.
The three exclusive groups are caller (caller marker anywhere in functions),
worker_temporally_associated (worker marker but no caller), and other. Retain
all groups, even empty ones. At least one positive caller alloc_bytes AND
alloc_objects contribution is required for this allocation-producing fixture.
This checks marker usability, not an exact allocation count.

Zero-active rows are a separate ordered list of snapshot+ordinal references,
not deduplicated and not assigned a size. Their zero numeric contribution is
excluded from the maps but their raw records remain in capture.json.

## Exact derived JSON schema

Top-level keys are exactly:

- schema: `"goai-control-alloc-sites-derived-v1"`.
- control: `"SoftplusF64_256K"`; n: 1024; gomaxprocs: actual capture value.
- caller_function, worker_function: the validated capture strings.
- windows: object with pre_snapshot, region, post_gc, post_snapshot,
  serialization. Each is a counters object with the same six keys as the input,
  but signed-64-bit differences. Pairs respectively are start-pre_raw_before,
  end-start, post_gc-end, post_raw-post_gc, tail-post_raw.
- raw_keys: array of the raw-key objects below.
- normalized_sites: array of the normalized objects below.
- zero_active: array of objects `{snapshot: "pre"|"post", ordinal: integer}`,
  in pre then post snapshot order and ascending ordinal within each.
- groups: object with caller, worker_temporally_associated, other. Each contains
  exactly pre, post, delta profile-counter objects, raw_key_count and
  normalized_site_count. Include all keys and all three groups, with zeros where
  appropriate. Counts refer to union contributors, not just nonzero deltas.
- profile_totals: exactly pre, post, delta profile-counter objects, summing every
  group, with checked arithmetic.
- immediate_minus_profile: exactly alloc_bytes, alloc_objects, free_objects.
  These are region.total_alloc-profile_totals.delta.alloc_bytes,
  region.mallocs-profile_totals.delta.alloc_objects, and
  region.frees-profile_totals.delta.free_objects respectively. There is no
  MemStats cumulative FreeBytes counter; do not invent a fourth discrepancy.
- interpretation: literal
  `"perturbed process-wide diagnostic; worker association only; no qualification"`.

Raw-key objects have exactly index, slot_size, pcs (nonzero prefix), functions
(ordered names), group, pre, post, delta, pre_ordinals, post_ordinals,
pre_cardinality, post_cardinality. Pre/post/delta are profile-counter objects.
Sort the raw-key union by `[slot_size, pcs]` using Ruby array/string comparison;
assign index consecutively from zero. Sort each ordinal list ascending, retaining
every row exactly once. Cardinalities are the respective ordinal-list lengths.
Raw PC lexical sorting is only deterministic output ordering, not normalization.

Normalized objects have exactly index, slot_size, functions, group, pre, post,
delta, raw_key_indices, raw_key_cardinality. Sort by `[slot_size, functions]`;
assign consecutive indices. Contributor indices are ascending; cardinality is
their count. Pre/post/delta sum those exact contributors. Do not encode composite
keys by ambiguous string concatenation. Empty lists are [], never null.

## Synthetic verification, before live use

Fixtures must be wholly synthetic and carry a positive caller contribution.
Independently expected values must exercise both collision layers, absent sides,
unchanged sites, zero-active multiplicity, changing PC/file/line values with
unchanged normalized identity, ordered/repeated/inline frames, caller precedence
over worker, worker-only and other groups, and positive/negative heap changes.
Use exact >2^53 values and near uint64 boundaries; deliberately test each signed
overflow path, malformed numeric identities, decreasing raw-key counters masked
by another increasing key, malformed stacks, unresolved active frames, false
snapshot status, prefix/capacity mismatch, every required scalar/environment
value, unknown/duplicate/missing keys at multiple depths and non-integer tokens.
Verify input immutability, deterministic serialization, exact contributor
coverage, global sums and successful strict JSON round trips. CLI tests cover
exclusive output, existing-output preservation, invalid input producing no
derived file, and nonzero errors. Pure tests only: no live profiling or matrix.

A fresh independent verifier must recompute the synthetic expectations without
treating the implementation transcript as proof. Runtime source, existing
rejections, prior measurements, and perfscan #968 claims remain unchanged.
