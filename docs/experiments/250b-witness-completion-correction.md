# 250b — qualify complete 250 arrays after a witness-count failure

250's intervention implementation, all 32 cases and all numerical installation,
replay and B/H identity checks completed. Its selector grouped by the entire
case `group` object (including assignment, block and target) rather than just
schema/order. It therefore saved witnesses for all 32 contexts at blocks
10, 19, 28, instead of the intended four contexts. Its final count assertion
expected 132 but observed 1,056 extra witness QK products and stopped before
writing status/seal. Preserve the original source, protocol, log, failure,
progress, every case array and metadata without modification.

The narrowly scoped correction reconstructs status from all 32 hash-bound case
records, each with all declared exact and tolerance checks passed. It checks
the actual all-context witness selection and exact observed budgets before
writing a completion receipt and final seal. No model is rerun, no case is
excluded, no outcome criterion or numerical tolerance changes. This completion
receipt distinguishes the failed original accounting from the qualified data.

The separate 250b auditor is the frozen 250 auditor with only witness coverage,
expected witness count and completion-receipt validation changed. It audits
all 1,056 saved QK/softmax/PV witnesses (all contexts, three fixed blocks,
11 arms); it is still not a separate reconstruction of every full forward or
every intervened block. All endpoint heads and original primary/secondary
criteria remain unchanged. No outcomes were scored to choose this correction.

`communication250b.py --recover --out EXISTING --input-242 PREREQUISITE` performs
this recovery only if the exact original count-only failure, complete progress
and absence of status/seal are present. On a new output, it runs the original
frozen producer, retains its expected accounting failure, then applies the
same bounded recovery. Any earlier or different failure is fatal. This explicit
wrapper preserves the original bytes and executable history.
