# M7 independently authored expectations

These are input recipes and expected results fixed before implementing the gdxrs
adapter. They are not snapshots generated from either backend. Tasks 042/043 must
create the native-written fixtures, retain generator/revision/input hashes, and
assert these literal expectations. Until then these are specifications, not tests
claimed to pass. Any native format normalization must be investigated and recorded;
do not rewrite expected results to match adapter output.

## Dictionary identity across batches

Register UELs in order `aa`, `bb`, `cc`, `dd` (IDs 1–4). Write parameter P with
ordered records `(1,10)`, `(2,20)`, `(3,30)`, `(4,40)` and request batch_size=2.

| Batch | Mode A enum labels | Mode A physical codes | Mode B UInt32 IDs | Float64 values |
| --- | --- | --- | --- | --- |
| 1 | aa, bb | 0, 1 | 1, 2 | 10, 20 |
| 2 | cc, dd | 2, 3 | 3, 4 | 30, 40 |

Every enum batch declares all four categories in registration order. Mode B
declares UInt32 in both batches. Repeated scans produce the same rows/schema;
retaining batch 1 while advancing/closing does not mutate its contents. A tiny
dictionary budget rejects mode A before output; mode B still streams without
collecting all labels. A global head(3) returns IDs 1,2,3, not three per batch.

## Label bytes

Register raw label bytes `61 61` and `ff 62` as IDs 1 and 2; write values 10,20.
Mode A fails with a label-encoding error before emitting output. Mode B returns
IDs `[1,2]` and values `[10,20]` without decoding labels. Its resolver called with
`[2,1,2]` returns rows `(2,b'\xffb')`, `(1,b'aa')`, `(2,b'\xffb')`, in that order,
with schema `{uel_id: UInt32, label: Binary}`. ID 0 or 3 rejects the whole request
with ValueError. UTF-8 string filter `aa` matches only ID 1. A request whose
conversion allocation exceeds admission fails before returning a partial table.

## Typed values and deliberate Float64 projection

Create a one-dimensional parameter with IDs 1–7 containing the following typed
values, in order. Register acronym ID 7 with an independently specified name.

| ID | Typed input | gdxrs Float64 expected | Legacy Float64 expected |
| --- | --- | --- | --- |
| 1 | EPS | +0.0, bits 0000000000000000 | +0.0 |
| 2 | UNDEF | null | NaN |
| 3 | NA | null | NaN |
| 4 | +INF | +infinity | +infinity |
| 5 | -INF | -infinity | -infinity |
| 6 | ordinary 1.25 | 1.25, bits 3ff4000000000000 | 1.25 |
| 7 | acronym 7 | null | native encoded acronym number, 7 × 10e300 |

With Preserve, gdxrs statistics report one EPS projection and one acronym
projection. ToNotAvailable still emits null for ID 7; it must not claim that an
ordinary numeric value was preserved. Check null validity separately from NaN:
IDs 2,3,7 have false validity bits; IDs 1,4,5,6 have true validity bits. Compare
backends on their explicitly different projections, not by treating null and NaN
as equal. Native typed inspection must identify the acronym, not infer it from
the new adapter's null output.

For adapter conversion unit tests, ordinary NumberBits inputs for +0 and -0
must produce bits 0000000000000000 and 8000000000000000 respectively. This is a
conversion expectation, not a claim that the native GDX writer retains signed
zero on disk. Core Exact conversion of typed EPS must continue to fail.

## Empty/scalar/domain metadata

- Empty one-dimensional parameter E has zero rows and key/value columns of the
  planned enum-or-UInt32/Float64 schema. A zero limit uses the same schema without
  decoding records or starting workers.
- Scalar parameter S=1.25 produces exactly one row and one `value: Float64`
  column, with no key columns. A scalar parameter Z with zero physically stored
  rows still produces one default +0.0 row, matching native GDX public semantics.
  A zero limit produces no rows for either scalar while preserving that schema.
- Register set I containing IDs 1,3, alias J pointing directly to I, and empty
  parameter D with strict domain J. Metadata must preserve J's target I and D's
  domain name J, rather than replacing D's domain with I. Domain enumeration of
  empty D contains no used labels. The universe enumerates all four registered
  UEL IDs in registration order, including IDs unused by I.

## Residuals, projection and lifecycle

Write ordered parameter records with IDs 1–4 and values `[1,10,2,20]`, batch_size=2.
`filter(value >= 10).head(2)` returns IDs `[2,4]` and values `[10,20]`. Projecting
only keys still retains the value internally until the residual has run. Unknown
or empty label filters return zero rows with the planned schema. Consuming one
batch and cancelling acknowledges exactly that batch; producer workers join and
temporary resources release. A failing sink or a source change after output
must raise with acknowledged progress and must never retry through legacy.

Frozen semantic fixtures cannot prove memory containment. Task 047 adds growing
record/metadata workloads under the contract caps, including an intentionally
exhausted scratch quota, a blocked bounded queue and cancellation/cleanup.
