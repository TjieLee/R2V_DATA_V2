"""Deterministic appearance-only diversity cues for pair replacement.

Cues are appearance hints for the Qwen replacement call only. They never add
objects, props or actions, and never touch the semantic identity: the same
input, seed and contract always produce the same two cues.
"""

import hashlib

GENERIC = "ordinary modern everyday appearance with no profession or period constraint; freely vary gender presentation and adult age band from the source"

# Everyday occupations only; each cue is expressed through clothing/grooming.
PROFESSION_CUES = (
    "doctor", "nurse", "pharmacist", "dentist", "paramedic", "veterinarian",
    "police officer", "security guard", "firefighter", "soldier", "sailor",
    "chef", "line cook", "baker", "barista", "waiter", "dishwasher",
    "teacher", "professor", "kindergarten teacher", "librarian", "student",
    "office professional", "bank clerk", "accountant", "lawyer", "receptionist",
    "programmer", "engineer", "technician", "electrician", "plumber", "mechanic",
    "carpenter", "welder", "construction worker", "miner", "factory worker",
    "farmer", "fisherman", "gardener", "florist", "butcher", "market vendor",
    "street food vendor", "supermarket cashier", "sales assistant", "delivery courier",
    "taxi driver", "bus driver", "truck driver", "train conductor", "pilot",
    "flight attendant", "tour guide", "hotel housekeeper", "hairdresser", "barber",
    "tailor", "shoemaker", "cleaner", "refuse collector", "postal worker",
    "photographer", "journalist", "musician", "painter", "dancer", "fitness coach",
    "coach", "social worker", "translator", "real estate agent", "insurance agent",
)

PERIOD_CUES = (
    "Qing-style noblewoman", "Qing-style nobleman", "traditional Chinese noblewoman",
    "ancient Chinese scholar", "Tang-style court attire", "Ming-style scholar attire",
    "hanfu-style figure", "traditional Chinese robe figure", "Republican-era formal attire",
    "1920s Shanghai-style attire", "vintage 1980s everyday attire", "retro stage attire",
)

# Ordinary clothing dominates; special cues stay a minority.
BUCKET_WEIGHTS = {"generic":65, "profession":25, "period":10}
BUCKETS = ("generic", "profession", "period")
DIVERSITY_CONTRACT = "pair_replacement_diversity_v2"


def _digest(*parts):
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def bucket_for_value(value):
    """Deterministic bucket mapping; unit-testable without sampling people."""
    cut = BUCKET_WEIGHTS["generic"]
    profession_cut = cut + BUCKET_WEIGHTS["profession"]
    if value < cut:
        return "generic"
    if value < profession_cut:
        return "profession"
    return "period"


def _bucket(seed, key, index, attempt=0):
    return bucket_for_value(int(_digest(DIVERSITY_CONTRACT, seed, key, index, attempt)[:8], 16) % 100)


def _cue(seed, key, index, attempt=0):
    name = _bucket(seed, key, index, attempt)
    if name == "generic":
        return GENERIC
    pool = PROFESSION_CUES if name == "profession" else PERIOD_CUES
    return pool[int(_digest(DIVERSITY_CONTRACT, seed, key, index, attempt)[8:16], 16) % len(pool)]


def diversity_cues(seed, row_sha256, *, index=0):
    """Two deterministic cues; specialized cues must not repeat each other."""
    first = _cue(seed,row_sha256,index)
    second = _cue(seed,row_sha256,index+1)
    attempt = 0
    while second != GENERIC and second == first and attempt < 8:
        attempt += 1
        second = _cue(seed,row_sha256,index+1,attempt)
    if second != GENERIC and second == first:
        second = GENERIC
    return first, second
