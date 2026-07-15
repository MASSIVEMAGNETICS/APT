# Migration from the original prototype

The original 168-line script demonstrated names and an interactive loop, but it
did not persist state, use its memory deque, create branches, learn from bytes, or
measure novelty and coherence from content. The optimization-labelled routine was
a root-finding variant invoked without a validated bracket.

APT v1 replaces those behaviors as follows:

| Prototype behavior | APT v1 replacement |
|---|---|
| Random novelty | Deterministic similarity-derived novelty |
| Fixed coherence | Defined context and continuity score |
| In-memory random-ID tree | Persistent hash-ID SQLite DAG |
| `fork()` printed a message | Branch head with preserved divergent children |
| Exit claimed persistence | Committed SQLite transactions and model archives |
| Unused memory deque | Immutable objects, occurrences and vector retrieval |
| Substring safety gate | Removed; no false safety claim |
| Unvalidated “optimizer” | Contracted bisection root finder |
| Canned responses | Locally trainable next-byte model and explicit candidate scoring |

The prototype remains recoverable from Git history. Keeping it in the active tree
would create two contradictory implementations, so the v1 branch removes it.

