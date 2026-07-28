"""Frozen experiment-level constants.

Values in this module are part of persisted fingerprints. Changing one must
create a new processed-data fingerprint and run identity.
"""

from __future__ import annotations

REPRESENTATION_VERSION = "MoleCode-TS/v1"
SIGNATURE_ALGORITHM = "rdkit-canonical-isomeric-mapless-directional-v1"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SEED = 42
MIN_TS_EDGE_WEIGHT = 0.1
MAX_SEQUENCE_LENGTH = 2048
MAX_NEW_TOKENS = 512
ALLOWED_BOND_ORDERS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
FORMAL_EVAL_K = (1, 2, 3, 4, 5, 10)

# Content identities of the only four files consumed by the frozen
# Transition1x preprocessing pipeline.  The recovery manifest supplies only
# corrected R/P mapped SMILES; raw pickles remain authoritative for splits and
# TS Wiberg bond orders.
TRANSITION1X_SOURCE_SHA256 = {
    "train.pkl": "2e879027934286f2ecf38a7807662f9ee7777c45f2b0af8f552083fc1c601686",
    "val.pkl": "39283bfa484b3c68a2138634a39ba91770f1b020349663066ba279004d47ada1",
    "test.pkl": "3c24fe33dacf38235f51dd01eb8c17aae2963f04294e060c08746cf99ef27e86",
    "recovery.jsonl": "5843fc724f4a0dff9b624617c383ed81532191e1de8bfaf21570e26ac9cf54ad",
}

QWEN_PAD_TOKEN_ID = 248044
QWEN_IM_START_TOKEN_ID = 248045
QWEN_IM_END_TOKEN_ID = 248046

SYSTEM_PROMPT = """MoleCode-TS/v1

Task: Predict the discretized transition-state molecular graph from the
reactant and product molecular graphs.

Shared atom IDs are fixed, zero-based, and identical in all states.
Hydrogen atoms are explicit. Input edge order is arbitrary. Reactant/product
state attributes and change sections are evidence, not output constraints.

Return exactly one final block:

<TS_EDGES>
aI --[bo=B]-- aJ
...
</TS_EDGES>

For every edge, I < J and B must be one of 0.5, 1, 1.5, 2, 2.5, or 3.
Emit every transition-state edge with B > 0. An unlisted atom pair means
B = 0. A transition-state edge may connect any two listed atoms, even when
that pair is absent from both reactant and product graphs. Sort edges by
(I, J).

Do not include transition-state stereochemistry, components, atom
attributes, comments, or any other text in the final answer."""

QUARANTINED_REACTION_IDS = (
    "rxn0951",
    "rxn1323",
    "rxn1324",
    "rxn1434",
    "rxn1889",
    "rxn3034",
    "rxn3760",
    "rxn4187",
    "rxn4998",
    "rxn5062",
    "rxn5063",
    "rxn5065",
    "rxn5570",
    "rxn7147",
    "rxn7475",
    "rxn9958",
)
