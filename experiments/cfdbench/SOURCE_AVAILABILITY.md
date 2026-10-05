# CFDBench source availability

This public repository includes the frozen CFDBench experiment protocol and the author-generated reference-result artifacts used by the paper.

The recovered private experiment implementation described its U-Net and FNO definitions as copies of pinned upstream CFDBench model semantics. During release preparation, the separate upstream source repository did not expose a root software license.

For that reason, those copied executable model definitions are **not redistributed here**.

This source-availability boundary does not change the reported result. The public repository retains:

- the frozen experiment protocol;
- the fixed representation metadata needed by the reported study;
- seedwise and family-level reference results;
- per-case error artifacts that were cleared for the public package.

Executable CFDBench model code can be added later only if redistribution rights are established or an independently licensed replacement is implemented and verified against the preserved result boundary.
