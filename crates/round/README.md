# infinitum-round

The speculative round as [infinitum](https://github.com/silvanshade-org/infinitum) describes it: a graph of typed fragments, the components that emit them, the contract a backend plans them under, and the host's decision between a round's execute and commit phases. Engine-neutral: no backend is linked here.

## The round

| Piece | Type | What it is |
| ----- | ---- | ---------- |
| Fragment | `FragmentKind` | What a node computes, at the granularity of model sub-blocks and round steps. The kind fixes its `Effect`, and the effect fixes its `Phase`. |
| Graph | `RoundGraph`, built by `RoundBuilder` | Fragments in insertion order, each reading earlier ones. The builder refuses an unknown or repeated input, an execute fragment after a commit fragment, a commit fragment that reads no acceptance, and an empty phase. |
| Components | `Drafter`, `Verifier`, `Acceptor`, `Committer`; `compose` | A method is one choice of the four. `dflash2(width)` composes DFlash2: context catch-up, block forward, proposal select, verify inputs, target forward, sparse rejection, then KV commit, GDN replay, feature publication. |
| Planning | `Backend`, `Refusal` | A backend plans a graph or refuses it, naming itself and the reason: no drafter it knows, the fragment where the graph leaves its known fusion, an unsupported width. |
| Host decision | `Preview`, `RoundOffer`, `RoundVerdict` | Each round's licensed tokens are offered before their commit. `Preview` admits them against the request's output budget, limits the round that would overrun it, and records every round. |

Edges carry data dependence only. Typed ports (dtype, shape envelope, sharding) arrive with the decomposed lowering that needs them; the first backend runs the whole DFlash2 round as one known fusion, so no port is read yet.
