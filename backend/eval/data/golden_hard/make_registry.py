"""
Generates docs/sodium_cell_registry.txt, the hard set's look-alike
distractor: 40 fictional sodium cells with names, suppliers, test sites and
numbers deliberately close to the real ones (KestrelPack-2 vs KestrelCell,
VX-11 vs VX-9, Oskarsund appears too). Retrieval that matches on surface
similarity pulls these in instead of the real evidence.

Deterministic (fixed seed), so the committed file can always be rebuilt:
    python eval/data/golden_hard/make_registry.py

Rules that keep every golden question answerable without ambiguity: no
entry is named KestrelCell or SodaCore exactly, none uses VX-9 or VX-9L,
and none mentions Norrby, Kade or the Halvik Institute.
"""

import random
from pathlib import Path

OUT = Path(__file__).parent / "docs" / "sodium_cell_registry.txt"

PREFIXES = ["KestrelPack", "Kestrel-S", "KestrelLite", "SodaCore-X", "SodaCore-Max", "FalconCell",
            "Merlin-Na", "HarrierCell", "OspreyPack", "SaltBank", "TernCell", "Gannet-S"]
ELECTROLYTES = [("VX-7", "Varn Materials"), ("VX-11", "Varn Materials"), ("NaFE-2", "Ostrand Chemicals"),
                ("Halyte-4", "Brix Electrolytes"), ("SolvaNa", "Ostrand Chemicals")]
SITES = ["Oskarsund Battery Test Centre", "Drammen Battery Lab", "Tromsdal Cell Laboratory", "the developer's own lab"]
DEVELOPERS = ["Ravnholt University", "Seaborg Energy", "Ostrand Chemicals", "Lysaker Institute",
              "Brix Electrolytes", "Nordhavn Technical University", "Tidewater Storage"]
CHEMISTRIES = ["room-temperature sodium-sulfur", "sodium-ion with a layered oxide cathode",
               "sodium-ion with a Prussian white cathode", "room-temperature sodium-sulfur"]


def main() -> None:
    rng = random.Random(20261104)
    names: set[str] = set()
    lines = [
        "Nordic Sodium Cell Registry, 2025 edition",
        "",
        "Compiled by the Nordic Battery Forum. The registry lists sodium-based cells whose developers "
        "submitted test data during 2024 and 2025. Figures are as reported by the developers and have "
        "not been independently verified unless an independent test site is named. Cycle life is given "
        "to 80 percent of initial capacity at a 1C rate unless stated otherwise.",
        "",
    ]
    while len(names) < 40:
        name = f"{rng.choice(PREFIXES)}-{rng.randint(1, 9)}" if rng.random() < 0.6 else rng.choice(PREFIXES)
        if name in names:
            continue
        names.add(name)
        electrolyte, supplier = rng.choice(ELECTROLYTES)
        cycles = rng.randrange(700, 3300, 10)
        energy = rng.randrange(120, 330, 5)
        cost = rng.randrange(48, 120)
        text = (
            f"Entry {len(names)}: {name}. Developer: {rng.choice(DEVELOPERS)}. Chemistry: {rng.choice(CHEMISTRIES)}. "
            f"Electrolyte: {electrolyte}, supplied by {supplier}. Tested at: {rng.choice(SITES)}. "
            f"Reported cycle life: {cycles:,} cycles. Cell-level energy density: {energy} watt-hours per kilogram. "
            f"Projected cost: {cost} US dollars per kilowatt-hour."
        )
        if rng.random() < 0.4:
            text += f" Calendar life: {rng.randint(8, 20)} years (developer estimate)."
        if rng.random() < 0.25:
            text += f" Capacity fade accelerates above {rng.choice([35, 40, 45, 50])} degrees Celsius."
        lines += [text, ""]
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(names)} entries)")


if __name__ == "__main__":
    main()
