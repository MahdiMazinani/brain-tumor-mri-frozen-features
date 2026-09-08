"""splits_v5.py -- deterministic split construction, identity grouping, and the
selection ledger used by code_final_fixed_v5.py.

Why this module exists
---------------------
The desk rejection named two defects. The second one is structural: "the final
test cohort must also stay fully independent of method development, not used for
training, validation, model or hyperparameter selection, or threshold
optimization". Prose cannot establish that. This module turns it into a
mechanism.

  * Every split is a *manifest*: an explicit, hashed list of files carrying a
    role (TRAIN / VALID / TEST), so what was fitted on what stays recoverable
    after the fact.
  * The TEST role is sealed. SelectionLedger refuses to authorize a read of a
    test cohort until the configuration has been frozen, and it counts every
    read that does happen -- so "read once" becomes an auditable number instead
    of a claim.
  * Identity grouping keeps byte-identical (optionally orientation-variant)
    copies of one image on a single side of a split, which plain stratified
    sampling does not.
  * Patient identity is resolved through PID *families*, because identifier
    strings differing only by a trailing letter (MR024780 / MR024780B) are not
    evidently different people. Identifier-disjoint is a weaker guarantee than
    person-disjoint, and the two should not be reported as if they were equal.

No torch and no sklearn here, so a split can be built, inspected and diffed on
any machine.
"""
from __future__ import annotations

import argparse
import collections
import glob as _glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

ROLES = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# Enumeration. The order here is load-bearing.
# ---------------------------------------------------------------------------
# code_final_fixed_v4.FolderDataset walks `for cls in class_names: for p in
# list_images(cls_dir)`, and list_images sorts os.listdir. Every cached .npy in
# ablation/caches/ therefore carries rows in (sorted class, sorted filename)
# order. Reproducing that order byte for byte is what lets v5 slice a validation
# split out of an existing train cache instead of paying for a second forward
# pass. Do not "improve" these two functions.

def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMAGE_EXTS)]


def discover_classes(split_dir: str) -> List[str]:
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"missing split folder: {split_dir}")
    out = [n for n in sorted(os.listdir(split_dir))
           if os.path.isdir(os.path.join(split_dir, n)) and list_images(os.path.join(split_dir, n))]
    if len(out) < 2:
        raise RuntimeError(f"need >= 2 populated class folders inside {split_dir}")
    return out


def enumerate_split(root: str, split: str, class_names: Sequence[str]) -> List[Tuple[str, str, int]]:
    """[(abs_path, class_name, row_index)] in cache row order for root/split."""
    out: List[Tuple[str, str, int]] = []
    i = 0
    for cls in class_names:
        for p in list_images(os.path.join(root, split, cls)):
            out.append((p, cls, i))
            i += 1
    if not out:
        raise RuntimeError(f"no images under {os.path.join(root, split)}")
    return out

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
# Two corpora, two situations.
#
# Cheng/figshare filenames carry the acquisition identifier before "__", e.g.
# 100820__209.png -> "100820", MR024780B__17.png -> "MR024780B". 166 of the 233
# identifiers are plain integers; 67 begin with "MR" and 40 of those differ from
# another identifier only by a trailing capital letter. MR024780B, MR024780D,
# MR024780E and MR024780G are four identifiers; whether they are four people is
# not something the filename can tell us. Grouping by identifier is therefore the
# weaker guarantee, and FAMILY grouping (strip one trailing letter) is the
# conservative one. v5 reports both instead of asserting either.
#
# The Kaggle four-class corpus carries no identifier at all, so the group unit
# there is the image, and identity can only be recovered from pixel content.

_PID_FAMILY_RE = re.compile(r"^(MR\d+)([A-Z])$")


def pid_from_path(p: str) -> str:
    """Acquisition identifier = filename stem up to '__'; stem otherwise."""
    base = str(p).replace("\\", "/").rstrip("/").split("/")[-1]
    stem = os.path.splitext(base)[0]
    return stem.split("__")[0] if "__" in stem else stem


def pid_family(pid: str) -> str:
    """MR024780B -> MR024780. Anything else is returned unchanged."""
    m = _PID_FAMILY_RE.match(pid)
    return m.group(1) if m else pid


def group_key(path: str, unit: str) -> str:
    """unit: 'image' | 'identifier' | 'family'."""
    if unit == "image":
        return os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0]
    pid = pid_from_path(path)
    if unit == "identifier":
        return pid
    if unit == "family":
        return pid_family(pid)
    raise ValueError(f"unknown grouping unit: {unit!r}")


# ---------------------------------------------------------------------------
# Content identity
# ---------------------------------------------------------------------------

def file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dihedral_key(path: str, side: int = 32, quantize: int = 4) -> str:
    """Orientation-invariant content key.

    A perceptual hash is not invariant to rotation or reflection, which is the
    blind spot the manuscript now admits. This reduces the image to a small
    grayscale grid, generates all eight dihedral transforms of that grid, and
    returns the lexicographically smallest quantized encoding. Two images that
    differ only by a flip and/or a multiple of 90 degrees collapse to the same
    key, so a duplicate audit over these keys catches mirrored copies that a
    byte-level or straight perceptual comparison misses.

    Requires Pillow. Deliberately not imported at module load, so that split
    construction works on a machine without it.
    """
    from PIL import Image  # local import on purpose

    with Image.open(path) as im:
        g = im.convert("L").resize((side, side), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    q = np.clip((a * (quantize - 1)).round(), 0, quantize - 1).astype(np.uint8)

    best: Optional[bytes] = None
    for base in (q, np.fliplr(q)):
        for r in range(4):
            cand = np.rot90(base, r).tobytes()
            if best is None or cand < best:
                best = cand
    return hashlib.md5(best).hexdigest()

# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

@dataclass
class SplitManifest:
    """One role of one bed: an explicit file list plus its provenance.

    `source_bed` / `source_splits` / `source_rows` are what make this cheap. A
    row is not re-extracted; it is *located*. Every row records which corpus
    folder its cached feature vector already lives in and at which index, so a
    validation split carved out of an existing 5,600-row train cache costs one
    fancy-index operation instead of a forward pass. It is also the audit trail:
    given a manifest you can point at the exact cache row behind any prediction.
    """
    bed: str
    role: str                       # train | valid | test
    root: str                       # corpus root the relative paths hang off
    class_names: List[str]
    files: List[str]                # relative to root, in manifest order
    labels: List[int]
    source_bed: str                 # bed key whose caches hold these features
    source_splits: List[str]        # on-disk folder per row (e.g. "Train")
    source_rows: List[int]          # row index within that folder's cache
    group_unit: str
    groups: List[str]
    recipe: Dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.files)

    def class_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {k: 0 for k in self.class_names}
        for y in self.labels:
            out[self.class_names[y]] += 1
        return out

    def n_groups(self) -> int:
        return len(set(self.groups))

    def digest(self) -> str:
        """Identity of the *content* of this split, order included."""
        h = hashlib.md5()
        for f, y in zip(self.files, self.labels):
            h.update(f.replace("\\", "/").encode("utf-8"))
            h.update(b"\x00")
            h.update(str(int(y)).encode("ascii"))
            h.update(b"\n")
        return h.hexdigest()

    def to_json(self) -> Dict[str, Any]:
        d = dict(bed=self.bed, role=self.role, root=self.root,
                 class_names=list(self.class_names), files=list(self.files),
                 labels=[int(v) for v in self.labels],
                 source_bed=self.source_bed,
                 source_splits=list(self.source_splits),
                 source_rows=[int(v) for v in self.source_rows],
                 group_unit=self.group_unit, groups=list(self.groups),
                 recipe=dict(self.recipe))
        d["n"] = self.n
        d["n_groups"] = self.n_groups()
        d["class_counts"] = self.class_counts()
        d["source_split_counts"] = dict(sorted(collections.Counter(self.source_splits).items()))
        d["digest"] = self.digest()
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "SplitManifest":
        m = SplitManifest(bed=d["bed"], role=d["role"], root=d["root"],
                          class_names=list(d["class_names"]), files=list(d["files"]),
                          labels=[int(v) for v in d["labels"]],
                          source_bed=d["source_bed"],
                          source_splits=list(d["source_splits"]),
                          source_rows=[int(v) for v in d["source_rows"]],
                          group_unit=d["group_unit"], groups=list(d["groups"]),
                          recipe=dict(d.get("recipe", {})))
        if "digest" in d and d["digest"] != m.digest():
            raise RuntimeError(
                f"manifest digest mismatch for {m.bed}/{m.role}: the stored file list "
                f"does not hash to the recorded digest. The split on disk was edited "
                f"after it was written; refusing to load it silently.")
        return m

    def abs_paths(self) -> List[str]:
        return [os.path.join(self.root, f) for f in self.files]

    def y(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=np.int64)

    def row_blocks(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """[(source_split, rows_in_that_cache, positions_in_this_manifest)].

        Feature assembly is then: allocate X of shape (n, D) and, for each block,
        X[positions] = cache[source_split][rows]. Order is preserved exactly.
        """
        by: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        for pos, (sp, r) in enumerate(zip(self.source_splits, self.source_rows)):
            by[sp].append((r, pos))
        out = []
        for sp in sorted(by):
            pairs = by[sp]
            out.append((sp,
                        np.asarray([r for r, _ in pairs], dtype=np.int64),
                        np.asarray([p for _, p in pairs], dtype=np.int64)))
        return out


@dataclass
class BedManifest:
    """All three roles of one bed, plus the audit the reviewer will ask for."""
    bed: str
    root: str
    class_names: List[str]
    group_unit: str
    seed: int
    splits: Dict[str, SplitManifest] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    built_at: str = ""
    builder: str = "splits_v5"

    def to_json(self) -> Dict[str, Any]:
        return dict(bed=self.bed, root=self.root, class_names=list(self.class_names),
                    group_unit=self.group_unit, seed=int(self.seed),
                    built_at=self.built_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                    builder=self.builder,
                    splits={k: v.to_json() for k, v in self.splits.items()},
                    audit=dict(self.audit))

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "BedManifest":
        return BedManifest(
            bed=d["bed"], root=d["root"], class_names=list(d["class_names"]),
            group_unit=d["group_unit"], seed=int(d["seed"]),
            splits={k: SplitManifest.from_json(v) for k, v in d["splits"].items()},
            audit=dict(d.get("audit", {})),
            built_at=d.get("built_at", ""), builder=d.get("builder", "splits_v5"))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "BedManifest":
        with open(path, "r", encoding="utf-8") as f:
            return BedManifest.from_json(json.load(f))

    def summary(self) -> str:
        L = [f"bed={self.bed}  group_unit={self.group_unit}  seed={self.seed}"]
        for role in ROLES:
            m = self.splits.get(role)
            if m is None:
                L.append(f"  {role:5s}: --")
                continue
            cc = m.class_counts()
            src = "+".join(sorted(set(m.source_splits)))
            L.append(f"  {role:5s}: n={m.n:5d}  groups={m.n_groups():4d}  "
                     + "  ".join(f"{k}={v}" for k, v in cc.items())
                     + f"   [{m.source_bed}/{src}, digest {m.digest()[:8]}]")
        ov = self.audit.get("group_overlap", {})
        if ov:
            L.append("  group overlap: " + "  ".join(f"{k}={v}" for k, v in sorted(ov.items())))
        d = self.audit.get("dedup")
        if d:
            # The removal count is the whole content of this bed, and --build/--show
            # print this summary and nothing else. Without these lines the only place
            # the number appears is the --dedup-mask path and the manifest JSON, so a
            # reader comparing nick against nick_dedup would see two beds that look
            # identical and no statement of what was taken out of which one.
            #
            # Built piecewise because manifests are also loaded from JSON written by
            # earlier versions: a key that is absent has to be omitted, not printed as
            # None. The fraction in particular is recomputed only when both operands
            # are there -- "(0.0%)" next to a missing pool would be a fabricated
            # number, which is the one thing an audit line must never be.
            bits = [f"removed {d['removed']}"] if "removed" in d else []
            if d.get("published_train_pool") is not None:
                bits.append(f"of {d['published_train_pool']} published train rows")
                if d.get("removed") is not None and d["published_train_pool"]:
                    bits.append(f"({100.0 * d['removed'] / d['published_train_pool']:.1f}%)")
            if d.get("removed_from"):
                bits.append(f"from {d['removed_from']}")
            if d.get("surviving_train_pool") is not None:
                bits.append(f"-- kept {d['surviving_train_pool']}")
            if bits:
                L.append("  dedup: " + " ".join(bits))
            if d.get("rule"):
                L.append(f"  dedup rule: {d['rule']}")
            pc = d.get("per_class_removed") or {}
            if pc:
                L.append("  dedup per class: "
                         + "  ".join(f"{k}={v}" for k, v in sorted(pc.items())))
            if self.audit.get("test_identical_to_nick"):
                L.append("  test cohort: byte-identical to bed 'nick' (digests agree), "
                         "so the two beds differ only in what was trained on")
        return "\n".join(L)


def group_overlap(manifests: Dict[str, SplitManifest]) -> Dict[str, int]:
    """Pairwise count of shared group keys. Every entry must be 0."""
    out: Dict[str, int] = {}
    keys = [r for r in ROLES if r in manifests]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            out[f"{a}_{b}"] = len(set(manifests[a].groups) & set(manifests[b].groups))
    return out


def assert_disjoint(manifests: Dict[str, SplitManifest]) -> Dict[str, int]:
    ov = group_overlap(manifests)
    bad = {k: v for k, v in ov.items() if v}
    if bad:
        raise RuntimeError(f"split groups overlap, which is exactly what this module exists "
                           f"to prevent: {bad}")
    return ov

# ---------------------------------------------------------------------------
# The selection ledger -- the mechanical answer to rejection ground (2)
# ---------------------------------------------------------------------------

SELECTABLE = (
    "feature_spec",         # which backbone combination
    "learner",              # svm / xgb / lgb / cat / ensemble
    "retained_variance",    # PCA target
    "standardization",
    "svm_C",
    "tree_rounds",
    "tta",
    "thresholds",           # per-class theta_c
    "ensemble_weights",     # beta on the simplex
    "decision_rule",        # argmax / thresholded / bayes / bayes_cal
    "cost_matrix",
    "temperature",
    "sample_weighting",
)


class TestCohortViolation(RuntimeError):
    """Raised when the test split is touched while selection is still open."""


class NonReportableBed(RuntimeError):
    """Raised when a demoted cohort is about to be treated as a result.

    Separate from TestCohortViolation because the two failures need different
    answers. A TestCohortViolation means the protocol was run out of order and
    something has to be undone; this means the protocol ran correctly on a cohort
    whose numbers cannot answer the comparability question, and the fix is either
    a different bed or an explicit statement that the run is diagnostic. Callers
    that catch one must not catch the other by accident, so neither inherits from
    the other. See assert_bed_reportable for the escape hatch.
    """


class SelectionLedger:
    """Records every selection decision, then seals the test cohort.

    The intended sequence is:

        led = SelectionLedger(run_dir)
        ...                                  # fit on train, score on valid
        led.record("retained_variance", 0.80, evidence="valid acc 96.1 vs 95.6 at 0.95")
        led.record("decision_rule", "bayes", evidence="valid risk 0.043 vs 0.061")
        led.freeze()                         # nothing further may be recorded
        Xte = led.authorize_test_read("bal", "test features")   # first read
        ...                                  # report

    Two properties follow, and both are checkable by someone who was not in the
    room. First, `record` after `freeze` raises, so a decision cannot be
    backdated. Second, `authorize_test_read` before `freeze` raises and after
    `freeze` increments a counter written into the ledger file, so the claim is
    not "we only looked once" but "the ledger says reads=1 and here it is".

    max_reads defaults to 1 per (bed, purpose). Legitimately reading the same
    cohort for several backbones is fine -- pass a distinct purpose, and the
    count per purpose stays visible.
    """

    def __init__(self, run_dir: str, name: str = "selection_ledger",
                 max_reads_per_purpose: int = 1, strict: bool = True):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, f"{name}.json")
        self.max_reads_per_purpose = int(max_reads_per_purpose)
        self.strict = bool(strict)
        self.decisions: List[Dict[str, Any]] = []
        self.reads: List[Dict[str, Any]] = []
        self.frozen_at: Optional[str] = None
        self.frozen_digest: Optional[str] = None
        self.notes: List[str] = []
        os.makedirs(run_dir, exist_ok=True)
        self._flush()

    # -- selection phase ---------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self.frozen_at is not None

    def record(self, knob: str, value: Any, evidence: str = "",
               split_used: str = "valid") -> None:
        if self.frozen:
            raise TestCohortViolation(
                f"selection is frozen (at {self.frozen_at}); cannot record {knob!r}={value!r}. "
                f"Anything decided after the test cohort is unsealed is a test-derived "
                f"choice and must be reported as such, not folded into the headline.")
        if split_used == "test":
            raise TestCohortViolation(
                f"refusing to record {knob!r} as chosen on the test split. That is the "
                f"exact practice the desk rejection named.")
        if knob not in SELECTABLE:
            self.notes.append(f"unlisted knob recorded: {knob}")
        self.decisions.append(dict(knob=knob, value=_jsonable(value), evidence=evidence,
                                   split_used=split_used,
                                   at=time.strftime("%Y-%m-%d %H:%M:%S")))
        self._flush()

    def record_many(self, mapping: Dict[str, Any], evidence: str = "",
                    split_used: str = "valid") -> None:
        for k, v in mapping.items():
            self.record(k, v, evidence=evidence, split_used=split_used)

    def freeze(self, note: str = "") -> str:
        """Seal the configuration. Returns the digest that pins it."""
        if self.frozen:
            return str(self.frozen_digest)
        payload = json.dumps([{k: d[k] for k in ("knob", "value")} for d in self.decisions],
                             sort_keys=True).encode("utf-8")
        self.frozen_digest = hashlib.md5(payload).hexdigest()
        self.frozen_at = time.strftime("%Y-%m-%d %H:%M:%S")
        if note:
            self.notes.append(note)
        self._flush()
        return self.frozen_digest

    # -- reporting phase ---------------------------------------------------
    def authorize_test_read(self, bed: str, purpose: str) -> None:
        if not self.frozen:
            raise TestCohortViolation(
                f"test cohort of bed {bed!r} requested for {purpose!r} before freeze(). "
                f"Fix the order: decide everything on validation, call freeze(), then read "
                f"the test split once.")
        key = f"{bed}:{purpose}"
        prior = sum(1 for r in self.reads if r["key"] == key)
        if prior >= self.max_reads_per_purpose:
            msg = (f"test cohort {bed!r} already read {prior} time(s) for purpose {purpose!r}; "
                   f"max_reads_per_purpose={self.max_reads_per_purpose}.")
            if self.strict:
                raise TestCohortViolation(msg)
            self.notes.append("WARNING: " + msg)
        self.reads.append(dict(key=key, bed=bed, purpose=purpose,
                               at=time.strftime("%Y-%m-%d %H:%M:%S")))
        self._flush()

    @classmethod
    def load(cls, run_dir: str, name: str = "selection_ledger",
             strict: bool = True) -> "SelectionLedger":
        """Reopen a ledger written by an earlier process.

        The selection stage and the reporting stage are separate invocations on
        purpose: whoever runs them can read the frozen configuration in between, and
        the process that finally touches the test cohort has no code path that could
        change a knob. That only works if reopening restores frozen_at -- otherwise
        the reporting stage could never obtain a test read at all.

        Reopening also re-hashes the decision list and compares it against the digest
        written at freeze time. If someone edits selection_ledger.json to swap in a
        better threshold after seeing the test numbers, the hashes disagree and this
        raises instead of quietly proceeding.
        """
        p = os.path.join(run_dir, f"{name}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"no ledger at {p}. Run the selection stage first: the test cohort "
                f"cannot be read without a frozen record of what was decided before it.")
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        led = cls(run_dir, name=name,
                  max_reads_per_purpose=int(d.get("max_reads_per_purpose", 1)),
                  strict=strict)
        led.decisions = list(d.get("decisions", []))
        led.reads = list(d.get("test_reads", []))
        led.frozen_at = d.get("frozen_at")
        led.frozen_digest = d.get("frozen_digest")
        led.notes = list(d.get("notes", []))
        if led.frozen_at:
            payload = json.dumps([{k: dd[k] for k in ("knob", "value")}
                                  for dd in led.decisions], sort_keys=True).encode("utf-8")
            recomputed = hashlib.md5(payload).hexdigest()
            if led.frozen_digest and recomputed != led.frozen_digest:
                raise TestCohortViolation(
                    f"the decisions in {p} no longer hash to the digest recorded at "
                    f"freeze time ({recomputed} vs {led.frozen_digest}). The frozen "
                    f"configuration was edited after sealing; the test numbers that "
                    f"follow would not be the ones the sealed configuration predicts.")
        led._flush()
        return led

    def read_counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in self.reads:
            c[r["key"]] = c.get(r["key"], 0) + 1
        return c

    # -- persistence -------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        return dict(frozen=self.frozen, frozen_at=self.frozen_at,
                    frozen_digest=self.frozen_digest,
                    max_reads_per_purpose=self.max_reads_per_purpose,
                    strict=self.strict, decisions=self.decisions,
                    test_reads=self.reads, read_counts=self.read_counts(),
                    notes=self.notes)

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        L = [f"selection ledger: {len(self.decisions)} decision(s), "
             f"frozen={'yes at ' + str(self.frozen_at) if self.frozen else 'NO'}"]
        for d in self.decisions:
            L.append(f"  {d['knob']:20s} = {d['value']!r:28s} on {d['split_used']:5s}"
                     + (f"   ({d['evidence']})" if d["evidence"] else ""))
        if self.frozen_digest:
            L.append(f"  config digest: {self.frozen_digest}")
        rc = self.read_counts()
        L.append("  test reads: " + (", ".join(f"{k}={v}" for k, v in sorted(rc.items()))
                                     if rc else "none"))
        for n in self.notes:
            L.append(f"  note: {n}")
        return "\n".join(L)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)

# ---------------------------------------------------------------------------
# Table I targets
# ---------------------------------------------------------------------------
# These are the numbers the manuscript already prints, kept here as the
# specification the builder must satisfy. If a rebuild cannot hit them the
# builder raises rather than silently reporting different counts than the paper.
#
# Balanced: 1,000 train + 400 valid + 400 test per class -> 4,000 / 1,600 / 1,600.
# On disk the Kaggle corpus ships Train=1,400 and Test=400 per class, so the
# validation split is exactly the 400-per-class carve-out of Train that leaves
# 1,000 for training. Nothing is invented and nothing is discarded.
#
# Imbalanced: the manuscript says "subsampling was applied to the training split
# alone". The corpus on disk contradicts that -- its Test folder holds
# 343/257/400/200 rather than 400 per class. The builder implements what the
# manuscript describes: train is subsampled, validation and test are the *same
# files* as the balanced bed. That also means the imbalanced bed needs no feature
# extraction of its own; every row is a row of a balanced-bed cache.

FOUR_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")

# WHERE EACH NUMBER BELOW COMES FROM -- read this before quoting any of them.
#
# bal: derived, not chosen. The corpus on disk ships 1,400 train + 400 test per
#      class, so 400-per-class validation carved out of Train leaves exactly
#      1,000. Every number follows from the folders plus one decision (how big
#      validation should be). It is still a non-reportable bed, for a different
#      reason -- the corpus is the published archive, but this re-cut of it is not
#      the published split, so no reader can reproduce the cohort. See
#      NON_REPORTABLE_BEDS, and use `nick` for the archive's own split.
#
# imb: INVENTED. 800/750/1000/500 appears in no published work and is derived
#      from nothing: it is a ratio someone picked, and the manuscript's own
#      claim that "subsampling was applied to the training split alone" is the
#      only thing it satisfies. It is retained solely so the retired imbalanced
#      tables remain reproducible and therefore explainable -- a figure you can
#      still recompute can be retired on the record, while a figure whose recipe
#      has been deleted can only be quietly dropped. Nothing may be reported
#      from it: rejection ground (1) asks for the cohorts and splits of prior
#      work, and a self-chosen subsample is the opposite of that.
#
#      The imbalance argument now rides on a corpus whose class prior is a
#      property of the data rather than of this file: figshare (1,426 / 708 / 930
#      slices, naturally uneven). The four-class archive cannot serve that
#      argument any more -- its current release is balanced at 1,400 per class --
#      so the point is made on figshare alone rather than on a hand-made ratio.
#
# The builder still asserts these counts. That is not an endorsement -- it is
# what makes "this cache is the imb cache" a checkable statement instead of a
# hope, which is precisely what verify_reproduction.py needs in order to certify
# the retired tables' provenance.

TARGETS: Dict[str, Dict[str, Dict[str, int]]] = {
    "bal": {
        "train": {"glioma": 1000, "meningioma": 1000, "notumor": 1000, "pituitary": 1000},
        "valid": {"glioma": 400, "meningioma": 400, "notumor": 400, "pituitary": 400},
        "test":  {"glioma": 400, "meningioma": 400, "notumor": 400, "pituitary": 400},
    },
    "imb": {
        # NO PUBLISHED BASIS -- see the block above. Diagnostic reproduction only.
        "train": {"glioma": 800, "meningioma": 750, "notumor": 1000, "pituitary": 500},
        "valid": {"glioma": 400, "meningioma": 400, "notumor": 400, "pituitary": 400},
        "test":  {"glioma": 400, "meningioma": 400, "notumor": 400, "pituitary": 400},
    },
}

PW_TARGETS = {"train": 1811, "valid": 472, "test": 781}

# ---------------------------------------------------------------------------
# Builder: the two four-class beds
# ---------------------------------------------------------------------------

CONTENT_GROUP_MODES = ("none", "md5", "dihedral", "strict")


def _union_find_groups(paths: Sequence[str],
                       relation: Dict[str, Sequence[str]]) -> Dict[str, str]:
    """Connected components of `relation` INDUCED on `paths`, keyed by member.

    Induced is the operative word: an edge is followed only when both of its ends
    are in `paths`. A training pool grouped this way never inherits a component
    through a file outside the pool -- a train row and a test row being the same
    picture is a leakage question, answered by removal and reported by
    strict_leakage_audit, and it must not quietly fuse two training groups.

    The key is the lexicographically smallest member rather than the union-find
    root, so it is a function of the component and not of the order the edges
    arrived in. _carve_validation sorts these keys and then shuffles them, so a key
    that moved with iteration order would move the validation cohort with it.
    """
    inside = set(paths)
    parent = {p: p for p in inside}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in inside:
        for q in relation.get(p, ()):
            if q in inside:                     # induced: skip edges leaving the pool
                a, b = find(p), find(q)
                if a != b:
                    parent[b] = a
    members: Dict[str, List[str]] = collections.defaultdict(list)
    for p in inside:
        members[find(p)].append(p)
    key_of = {r: min(v) for r, v in members.items()}
    return {p: key_of[find(p)] for p in inside}


def _content_groups(paths: Sequence[str], mode: str,
                    relation: Optional[Dict[str, Sequence[str]]] = None
                    ) -> Dict[str, str]:
    """path -> content group key. mode: 'none' | 'md5' | 'dihedral' | 'strict'.

    'md5' keeps byte-identical copies together; 'dihedral' additionally keeps
    flipped/rotated copies together. Grouping by content matters because the
    Kaggle corpus has 187 redundant byte-identical copies inside Train, and a
    naive per-class shuffle can put one copy in train and its twin in validation,
    which quietly inflates the validation score the whole selection rests on.

    'strict' is the only mode that is not a per-file key function. It groups by
    connected components of the DEDUP_RULE_ID relation supplied in `relation`
    (path -> its strict duplicate partners), which is a set-level property: A~B and
    B~C put A and C in one group even when A and C do not pair directly. It exists
    because md5 and dihedral both under-group here. Measured on the published
    four-class Train folder: md5 grouping leaves 694 train-validation strict
    duplicate pairs in bed `nick` and 372 in `nick_dedup`; 'strict' leaves 0 of
    each, at identical split sizes (4760/840 and 3661/647). A validation row that
    is the same picture as a training row inflates precisely the score that selects
    the configuration, the weights and the thresholds, so it is the same defect as
    test leakage one level down.

    `relation` is required for 'strict' and ignored otherwise -- passing it is the
    caller's way of saying it has already paid for the pixel pass, since building
    the relation costs a corpus decode and this function must stay free to call.
    """
    if mode == "strict":
        if relation is None:
            raise ValueError(
                "content_group='strict' needs the strict duplicate relation; call "
                "strict_duplicate_relation(...) (or pass content_group='md5' to group "
                "byte-identical copies only)")
        return _union_find_groups(paths, relation)
    if relation is not None and mode != "strict":
        raise ValueError(f"content_group={mode!r} takes no relation argument; only "
                         f"'strict' uses one")
    if mode == "none":
        return {p: p for p in paths}
    if mode not in ("md5", "dihedral"):
        raise ValueError(f"unknown content_group mode {mode!r}; "
                         f"expected one of {CONTENT_GROUP_MODES}")
    fn = file_md5 if mode == "md5" else dihedral_key
    return {p: fn(p) for p in paths}


def _strict_relation_for(root: str, mode: str, out_dir: Optional[str],
                         folders: Sequence[str], classes: Sequence[str],
                         verbose: bool = True) -> Optional[Dict[str, List[str]]]:
    """The strict duplicate relation when mode needs it, None otherwise.

    One line at every carve site instead of an if/else, so a builder cannot support
    the mode in its signature and then silently ignore it. The pair set is cached on
    disk per corpus folder, so the second builder to ask pays nothing.
    """
    if mode != "strict":
        return None
    rep = strict_duplicate_pairs(root, class_names=list(classes), folders=tuple(folders),
                                 cache_path=strict_pairs_path(root, out_dir),
                                 verbose=verbose)
    return strict_duplicate_relation(rep, root=root)


def build_four_class_beds(
    bal_root: str,
    seed: int = 2026,
    valid_per_class: int = 400,
    imb_train: Optional[Dict[str, int]] = None,
    content_group: str = "md5",
    strict_targets: bool = True,
) -> Dict[str, BedManifest]:
    """Rebuild both four-class beds from the balanced corpus alone.

    Returns {"bal": BedManifest, "imb": BedManifest}.

    Both beds draw validation from the balanced Train folder and share the
    balanced Test folder untouched, so:
      * no folder on disk is moved, copied or deleted -- the manifest is the split;
      * the imbalanced bed differs from the balanced one only in its training
        rows, which is what the manuscript claims and what makes the 4,000 -> 3,050
        comparison a controlled one;
      * every row of both beds is already covered by the existing balanced-bed
        caches, so no re-extraction is needed for train/valid/test.
    """
    imb_train = dict(imb_train or TARGETS["imb"]["train"])
    classes = discover_classes(os.path.join(bal_root, "Train"))
    if tuple(classes) != FOUR_CLASSES:
        raise RuntimeError(f"expected classes {FOUR_CLASSES}, found {tuple(classes)}")

    tr = enumerate_split(bal_root, "Train", classes)
    te = enumerate_split(bal_root, "Test", classes)

    # ---- content groups over the training pool -------------------------------
    # Deliberately NOT offered 'strict' here, and this is a decision rather than an
    # omission. Both beds this function builds are in NON_REPORTABLE_BEDS: `bal` is
    # not the published split and `imb` is a construction of ours, so neither number
    # reaches the manuscript and tightening their carve would buy nothing reportable.
    # Against that, every existing feature cache was extracted against these two
    # manifests; re-grouping them would move rows between train and valid and
    # invalidate the caches, costing a full re-extraction for beds that cannot be
    # cited. The strict relation is wired into the two beds where it pays --
    # build_published_four_class_bed and the deduplicated bed -- and _content_groups
    # will raise here if 'strict' is ever passed without a relation, which is the
    # behaviour we want: loud, not silently degraded to md5.
    cg = _content_groups([p for p, _, _ in tr], content_group)

    # ---- per class, shuffle content groups and cut validation ----------------
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[Tuple[str, int]]] = {c: [] for c in classes}
    for p, c, i in tr:
        by_class[c].append((p, i))

    valid_sel: Dict[str, List[Tuple[str, int]]] = {}
    train_pool: Dict[str, List[Tuple[str, int]]] = {}
    audit_groups: Dict[str, Any] = {}

    for c in classes:
        items = by_class[c]
        buckets: Dict[str, List[Tuple[str, int]]] = collections.defaultdict(list)
        for p, i in items:
            buckets[cg[p]].append((p, i))
        keys = sorted(buckets)
        rng.shuffle(keys)

        picked: List[Tuple[str, int]] = []
        used: set = set()
        for k in keys:
            if len(picked) >= valid_per_class:
                break
            grp = buckets[k]
            if len(picked) + len(grp) > valid_per_class:
                continue          # never split a content group across the cut
            picked.extend(grp)
            used.add(k)
        if len(picked) != valid_per_class:
            # fall back to a size-aware pass so exact counts stay reachable
            for k in sorted(keys, key=lambda kk: len(buckets[kk])):
                if len(picked) >= valid_per_class:
                    break
                if k in used:
                    continue
                grp = buckets[k]
                if len(picked) + len(grp) > valid_per_class:
                    continue
                picked.extend(grp)
                used.add(k)
        if len(picked) != valid_per_class and strict_targets:
            raise RuntimeError(
                f"class {c}: could only place {len(picked)} of {valid_per_class} validation "
                f"images without splitting a content group ({len(keys)} groups available)")

        rest = [(p, i) for k in keys if k not in used for p, i in buckets[k]]
        valid_sel[c] = sorted(picked, key=lambda t: t[1])
        train_pool[c] = sorted(rest, key=lambda t: t[1])
        audit_groups[c] = dict(n_content_groups=len(keys),
                               n_multi_copy_groups=sum(1 for k in keys if len(buckets[k]) > 1),
                               n_valid_groups=len(used))

    # ---- assemble the manifests ---------------------------------------------
    cls_idx = {c: i for i, c in enumerate(classes)}
    rel = lambda p: os.path.relpath(p, bal_root).replace("\\", "/")

    def mk(bed: str, role: str, sel: Dict[str, List[Tuple[str, int]]],
           src_split: str, recipe: Dict[str, Any]) -> SplitManifest:
        files, labels, rows, srcs, groups = [], [], [], [], []
        for c in classes:                       # class-major order, as the cache is
            for p, i in sel[c]:
                files.append(rel(p))
                labels.append(cls_idx[c])
                rows.append(i)
                srcs.append(src_split)
                groups.append(group_key(p, "image"))
        return SplitManifest(bed=bed, role=role, root=bal_root, class_names=list(classes),
                             files=files, labels=labels, source_bed="bal",
                             source_splits=srcs, source_rows=rows,
                             group_unit="image", groups=groups, recipe=recipe)

    test_sel: Dict[str, List[Tuple[str, int]]] = {c: [] for c in classes}
    for p, c, i in te:
        test_sel[c].append((p, i))

    base_recipe = dict(seed=int(seed), content_group=content_group,
                       valid_per_class=int(valid_per_class),
                       source_corpus=os.path.basename(os.path.abspath(bal_root)))

    out: Dict[str, BedManifest] = {}

    # -- balanced bed ---------------------------------------------------------
    bal_train = {c: train_pool[c] for c in classes}
    bal = BedManifest(bed="bal", root=bal_root, class_names=list(classes),
                      group_unit="image", seed=int(seed),
                      built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    bal.splits["train"] = mk("bal", "train", bal_train, "Train",
                             dict(base_recipe, rule="balanced Train minus the validation carve-out"))
    bal.splits["valid"] = mk("bal", "valid", valid_sel, "Train",
                             dict(base_recipe, rule=f"{valid_per_class}/class carved from Train, "
                                                    f"content groups kept intact"))
    bal.splits["test"] = mk("bal", "test", test_sel, "Test",
                            dict(base_recipe, rule="balanced Test folder, untouched"))
    out["bal"] = bal

    # -- imbalanced bed: subsample TRAIN ONLY ---------------------------------
    # Nested subsampling: the smaller training set is a subset of the larger one,
    # so the balanced-vs-imbalanced contrast isolates class prior and training
    # size rather than mixing in a different draw of images.
    rng2 = np.random.default_rng(seed + 1)
    imb_train_sel: Dict[str, List[Tuple[str, int]]] = {}
    for c in classes:
        want = int(imb_train[c])
        pool = list(bal_train[c])
        if want > len(pool):
            raise RuntimeError(f"imbalanced target for {c} is {want} but only {len(pool)} "
                               f"training images remain after the validation carve-out")
        idx = np.arange(len(pool))
        rng2.shuffle(idx)
        keep = sorted(idx[:want].tolist())
        imb_train_sel[c] = [pool[j] for j in keep]

    imb = BedManifest(bed="imb", root=bal_root, class_names=list(classes),
                      group_unit="image", seed=int(seed),
                      built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    imb.splits["train"] = mk("imb", "train", imb_train_sel, "Train",
                             dict(base_recipe, rule="nested subsample of the balanced train split",
                                  per_class_target=dict(imb_train)))
    imb.splits["valid"] = mk("imb", "valid", valid_sel, "Train",
                             dict(base_recipe, rule="identical to the balanced validation split"))
    imb.splits["test"] = mk("imb", "test", test_sel, "Test",
                            dict(base_recipe, rule="identical to the balanced test split"))
    out["imb"] = imb

    # ---- audits --------------------------------------------------------------
    for key, bm in out.items():
        ov = assert_disjoint(bm.splits)
        bm.audit["group_overlap"] = ov
        bm.audit["content_groups_per_class"] = audit_groups
        bm.audit["content_group_mode"] = content_group
        bm.audit["file_overlap"] = {
            f"{a}_{b}": len(set(bm.splits[a].files) & set(bm.splits[b].files))
            for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
        }
        bm.audit["targets"] = TARGETS[key]
        bm.audit["target_match"] = {
            role: (bm.splits[role].class_counts() == TARGETS[key][role]) for role in ROLES
        }
        bad = [r for r, ok in bm.audit["target_match"].items() if not ok]
        if bad and strict_targets:
            got = {r: bm.splits[r].class_counts() for r in bad}
            raise RuntimeError(f"bed {key}: split(s) {bad} do not match Table I. "
                               f"got={got} want={ {r: TARGETS[key][r] for r in bad} }")

    # cross-bed relationship, so the nesting claim is checkable
    nested = set(out["imb"].splits["train"].files) <= set(out["bal"].splits["train"].files)
    for bm in out.values():
        bm.audit["imb_train_nested_in_bal_train"] = bool(nested)
        bm.audit["valid_identical_across_beds"] = (
            out["imb"].splits["valid"].digest() == out["bal"].splits["valid"].digest())
        bm.audit["test_identical_across_beds"] = (
            out["imb"].splits["test"].digest() == out["bal"].splits["test"].digest())
    if not nested:
        raise RuntimeError("the imbalanced training split is not a subset of the balanced one; "
                           "the controlled-contrast claim would be false")
    return out

# ---------------------------------------------------------------------------
# Builder: the published four-class corpus, used with its published split
# ---------------------------------------------------------------------------
# Rejection ground (1) asks for a comparison "under directly comparable
# conditions: the same cohorts, training and validation splits, preprocessing,
# evaluation criteria, and held-out test cases". That needs a cohort published
# work also reports on, and the corpus in `balance 4class` IS one: it is the
# current release of the Kaggle four-class archive, unpacked verbatim.
#
# An earlier version of this comment said the opposite -- that 1,800 images per
# class made the folder a re-balanced local copy of a 1,621/1,645/2,000/1,757
# original. That was wrong about which release. The archive was updated and now
# ships exactly 1,800 per class, 1,400 Train + 400 Test, 7,200 files; the
# 7,023-file layout those four numbers describe is the superseded release, which
# Kaggle no longer serves and which is not obtainable any more. Evidence that the
# folder is an extraction rather than something built here: all 7,200 mtimes fall
# inside one two-minute window, the `Tr-`/`Te-` prefixes are the archive's own,
# and the JPEG quantization tables span 50 encoder families -- the signature of a
# three-source merge (figshare + SARTAJ + Br35H), not of a local re-save, which
# would stamp one table on every file it wrote.
#
# So the published folders are adopted verbatim, which is the whole point: the
# held-out cohort is not ours to choose, so it cannot have been chosen to flatter
# us. Only the validation cohort is ours, and it is carved out of Train -- there
# is no valid/ folder to use instead.
#
# What the folder being authentic does NOT buy is a clean held-out claim. About
# 70% of the archive's own real test files are the same picture as one of its own
# real training files, usually mirrored (glioma 117/400, meningioma 270/297,
# notumor 360/400, pituitary 295/400; verified at 128, 256 and native resolution,
# with 25 meningioma pairs identical pixel for pixel). That overlap arrives with
# the archive -- inside a verified pair the two halves carry quantization tables
# from two different pre-existing families in opposite order, and in pituitary not
# one of the 295 pairs shares a table -- so it is a property of the published
# split, not a defect of this project, and no re-download removes it. It is why
# `nick_dedup` exists beside `nick`: see build_deduplicated_four_class_bed.
#
# Per-class counts are still asserted, not assumed. They are the fingerprint that
# pins which release is on disk, so a future re-extraction of a different one
# fails loudly instead of silently changing what the reported number means.

# The current release: 1,400 per class in Train, 400 per class in Test. These are
# folder counts, so they include the 203 files the archive itself names as
# augmentations -- 100 in Train/meningioma and 103 in Test/meningioma, the only
# class that has any. Those files are part of the published cohort and are kept:
# deleting them would leave meningioma at 1,300/297, which is nobody's published
# number, and would trade a stated limitation for an unstated deviation.
NICK_PUBLISHED: Dict[str, Dict[str, int]] = {
    "train": {"glioma": 1400, "meningioma": 1400, "notumor": 1400, "pituitary": 1400},
    "test":  {"glioma": 400,  "meningioma": 400,  "notumor": 400,  "pituitary": 400},
}

# The superseded layout, kept so that a re-extraction of the older archive is
# recognised and named rather than merely rejected. Pass it as `published=` to
# build against that release if a copy ever turns up.
NICK_PUBLISHED_V1: Dict[str, Dict[str, int]] = {
    "train": {"glioma": 1321, "meningioma": 1339, "notumor": 1595, "pituitary": 1457},
    "test":  {"glioma": 300,  "meningioma": 306,  "notumor": 405,  "pituitary": 300},
}

# Spellings the archive is distributed under. Order is preference, not priority:
# the folder that exists wins, and its real on-disk spelling is what the manifest
# records, because that word is what the cache filename is keyed on.
CORPUS_FOLDERS: Dict[str, Tuple[str, ...]] = {
    "train": ("Train", "Training"),
    "valid": ("Valid", "Validation"),
    "test":  ("Test", "Testing"),
}


def resolve_corpus_folder(root: str, role: str,
                          candidates: Optional[Sequence[str]] = None) -> str:
    """The real on-disk spelling of `root`'s folder for `role`, or a loud failure.

    Every folder probe in this project used to be `os.path.isdir(root + name)`
    over a hardcoded ("Train", "Test"), and `list_images` returns [] for a missing
    folder. A corpus shipped as Training/ + Testing/ therefore enumerated as zero
    images and still wrote a completed-looking manifest. This returns the actual
    spelling -- matched case-insensitively against the directory listing, so the
    word stored in the manifest is the word on disk even on a case-insensitive
    filesystem -- and raises when nothing matches.
    """
    cands = tuple(candidates or CORPUS_FOLDERS[role])
    if not os.path.isdir(root):
        raise FileNotFoundError(f"corpus root does not exist: {root}")
    by_lower = {e.lower(): e for e in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, e))}
    for c in cands:
        hit = by_lower.get(c.lower())
        if hit is not None:
            return hit
    raise FileNotFoundError(
        f"no {role!r} folder under {root}: looked for {list(cands)}, "
        f"found subfolders {sorted(by_lower.values())}")


def assert_extraction_ready(root: str, train_folder: str, test_folder: str) -> None:
    """Refuse folder names the v4 extractor cannot serve, before any GPU time.

    Two independent failures, both silent, both discovered only after hours:

      * PATH. code_final_fixed_v4 opens ``os.path.join(cfg.data_dir, split)`` with
        the lowercase split word, and extract_caches_v2 hands it 'train'/'test'
        because that word is also the cache-filename key. Windows matches case
        insensitively, so 'train' opens Train/ -- but 'Training' is a different
        word, not a different case, and would not be found at all.
      * AUGMENTATION. v4:446 is ``tta_n = cfg.tta if split in ["valid","test"]
        else 1``. Under a folder called Testing/ the tta=3 extraction would write
        a byte-copy of the tta=1 one, CacheIndex would see one representative
        instead of a pair, and the split would come back `undetermined`.

    So the archive's Training/ + Testing/ have to be renamed. That is one command
    each, and it is far cheaper than discovering either failure downstream.
    """
    bad = [(w, r) for w, r in ((train_folder, "train"), (test_folder, "test"))
           if str(w).strip().lower() != r]
    if not bad:
        return
    cmds = "\n".join(f"    ren \"{os.path.join(root, w)}\" {r.capitalize()}" for w, r in bad)
    raise RuntimeError(
        f"{root} uses folder name(s) {[w for w, _ in bad]}, which the v4 extractor "
        f"cannot serve. Rename them first:\n{cmds}\n"
        f"Reason: v4 opens data_dir/<lowercase split word> (so 'Training' is never "
        f"found) and augments only when that word is 'valid' or 'test' (so a "
        f"'Testing' folder would make the tta=1 and tta=3 caches identical and the "
        f"TTA pair could never form).")


def _carve_validation(by_class: Dict[str, List[Tuple[str, int]]],
                      cg: Dict[str, str],
                      want: Dict[str, int],
                      rng: "np.random.Generator",
                      strict: bool = True):
    """Per class, cut `want[c]` validation rows without splitting a content group.

    Same mechanics as the balanced-bed carve, with a per-class target instead of
    one number, so a corpus whose classes are unequal keeps its class prior on
    both sides of the cut. build_four_class_beds deliberately keeps its own inline
    copy: its manifests are already extracted against, and a shared helper that
    changed their row order by one shuffle call would invalidate every cache.
    """
    valid_sel: Dict[str, List[Tuple[str, int]]] = {}
    train_pool: Dict[str, List[Tuple[str, int]]] = {}
    audit: Dict[str, Any] = {}
    for c in sorted(by_class):
        n_want = int(want[c])
        buckets: Dict[str, List[Tuple[str, int]]] = collections.defaultdict(list)
        for p, i in by_class[c]:
            buckets[cg[p]].append((p, i))
        keys = sorted(buckets)
        rng.shuffle(keys)

        picked: List[Tuple[str, int]] = []
        used: set = set()
        for k in keys:
            if len(picked) >= n_want:
                break
            grp = buckets[k]
            if len(picked) + len(grp) > n_want:
                continue          # never split a content group across the cut
            picked.extend(grp)
            used.add(k)
        if len(picked) != n_want:
            for k in sorted(keys, key=lambda kk: len(buckets[kk])):
                if len(picked) >= n_want:
                    break
                if k in used:
                    continue
                grp = buckets[k]
                if len(picked) + len(grp) > n_want:
                    continue
                picked.extend(grp)
                used.add(k)
        if len(picked) != n_want and strict:
            raise RuntimeError(
                f"class {c}: could only place {len(picked)} of {n_want} validation "
                f"images without splitting a content group ({len(keys)} groups available)")

        rest = [(p, i) for k in keys if k not in used for p, i in buckets[k]]
        valid_sel[c] = sorted(picked, key=lambda t: t[1])
        train_pool[c] = sorted(rest, key=lambda t: t[1])
        audit[c] = dict(n_content_groups=len(keys),
                        n_multi_copy_groups=sum(1 for k in keys if len(buckets[k]) > 1),
                        n_valid_groups=len(used), n_valid=len(picked),
                        n_train=len(rest), target=n_want)
    return valid_sel, train_pool, audit


def build_published_four_class_bed(
    root: str,
    bed: str = "nick",
    seed: int = 2026,
    valid_frac: float = 0.15,
    content_group: str = "md5",
    published: Optional[Dict[str, Dict[str, int]]] = None,
    strict_published: bool = True,
    strict_targets: bool = True,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> BedManifest:
    """Manifest the published four-class corpus on its own published split.

    The evaluation folder is adopted verbatim and never touched: it is the split
    the comparison literature reports on, which is the point -- a held-out cohort
    we did not choose cannot have been chosen to flatter the method, and that is
    rejection ground (2) answered by construction rather than by promise.

    Validation is carved out of the training folder, per class, at `valid_frac` of
    each class's training count, with content groups kept intact. Per class rather
    than globally so the class prior is the same on both sides of the cut; the
    corpus is unbalanced by design and re-balancing it here would recreate exactly
    the artefact that makes `balance 4class` unusable.

    content_group='strict' keeps whole connected components of the DEDUP_RULE_ID
    relation on one side of that cut, which is the only mode that leaves no
    validation row a duplicate of a training row: measured on this corpus, md5
    grouping leaves 694 such pairs and 'strict' leaves 0, at the same 4760/840
    sizes. It costs one cached pixel pass over the corpus (`out_dir` says where the
    cache goes); the other modes cost nothing.

    `published` is a fingerprint, not a target: the counts are compared against the
    folders and a mismatch raises, because the entire same-cohort claim rests on
    this folder being the published corpus rather than a re-balanced copy of it.
    Pass ``strict_published=False`` to record the deviation and continue.

    Returns one BedManifest whose three roles are train / valid / test. Nothing on
    disk moves, so a later re-run with a different seed re-cuts validation out of
    the same cached rows without a second forward pass.
    """
    pub = published or NICK_PUBLISHED
    tr_folder = resolve_corpus_folder(root, "train")
    te_folder = resolve_corpus_folder(root, "test")
    assert_extraction_ready(root, tr_folder, te_folder)

    classes = discover_classes(os.path.join(root, tr_folder))
    if tuple(classes) != FOUR_CLASSES:
        raise RuntimeError(f"expected classes {FOUR_CLASSES} under {root}/{tr_folder}, "
                           f"found {tuple(classes)}")
    te_classes = discover_classes(os.path.join(root, te_folder))
    if tuple(te_classes) != tuple(classes):
        raise RuntimeError(f"{root}: {tr_folder} holds {tuple(classes)} but {te_folder} "
                           f"holds {tuple(te_classes)}; the two folders must agree")

    tr = enumerate_split(root, tr_folder, classes)
    te = enumerate_split(root, te_folder, classes)

    # ---- provenance: is this the published corpus? ---------------------------
    got = {"train": collections.Counter(c for _, c, _ in tr),
           "test":  collections.Counter(c for _, c, _ in te)}
    prov: Dict[str, Any] = {"folders": {"train": tr_folder, "test": te_folder},
                            "published": {k: dict(v) for k, v in pub.items()},
                            "on_disk": {k: {c: int(v[c]) for c in classes}
                                        for k, v in got.items()},
                            "matches_published": {}}
    for role in ("train", "test"):
        want = pub.get(role, {})
        prov["matches_published"][role] = all(
            int(got[role][c]) == int(want.get(c, -1)) for c in classes)
    prov["augmented_filenames"] = {
        role: sum(1 for p, _, _ in (tr if role == "train" else te)
                  if "aug" in os.path.basename(p).lower())
        for role in ("train", "test")}
    bad = [r for r, ok in prov["matches_published"].items() if not ok]
    if bad and strict_published:
        alt = ("  (this looks like the superseded 7,023-file release; pass "
               "published=NICK_PUBLISHED_V1)\n"
               if all(int(got[r][c]) == int(NICK_PUBLISHED_V1.get(r, {}).get(c, -1))
                      for r in bad for c in classes) else "")
        raise RuntimeError(
            f"bed {bed}: folder(s) {bad} do not match the published per-class counts, so "
            f"this corpus is not the one the comparison literature reports on.\n"
            f"  on disk:   { {r: prov['on_disk'][r] for r in bad} }\n"
            f"  published: { {r: pub[r] for r in bad} }\n"
            f"{alt}"
            f"Re-extract the archive, or pass strict_published=False to record the "
            f"deviation and continue -- in which case no same-cohort comparison may be "
            f"claimed for this bed.")
    # The 103 augmentation-named files in Test/meningioma are the archive's own, so
    # they are recorded and kept rather than treated as a fault. An earlier version
    # raised here, on the rule that a sealed cohort must be acquired images. The rule
    # is right in general and wrong here: this cohort is not ours to curate, and
    # dropping those files would put meningioma at 297 test images, a count no
    # published work reports, which forfeits the same-cohort comparison the bed
    # exists to make. They are counted in the manifest, surfaced in the audit, and
    # named in the limitations -- a stated deviation from ideal beats a silent one.
    # `nick_dedup` is where the leakage question is actually answered.
    prov["augmented_in_test_kept"] = int(prov["augmented_filenames"]["test"])

    # ---- carve validation out of the training folder -------------------------
    cg = _content_groups([p for p, _, _ in tr], content_group,
                         relation=_strict_relation_for(
                             root, content_group, out_dir,
                             (tr_folder, te_folder), classes, verbose=verbose))
    by_class: Dict[str, List[Tuple[str, int]]] = {c: [] for c in classes}
    for p, c, i in tr:
        by_class[c].append((p, i))
    if not 0.0 < float(valid_frac) < 0.5:
        raise ValueError(f"valid_frac must lie in (0, 0.5), got {valid_frac!r}")
    want = {c: int(round(len(by_class[c]) * float(valid_frac))) for c in classes}
    thin = {c: n for c, n in want.items() if n < 2}
    if thin:
        raise RuntimeError(f"valid_frac={valid_frac} leaves {thin} validation images; "
                           f"raise it or use a larger corpus")

    rng = np.random.default_rng(seed)
    valid_sel, train_pool, group_audit = _carve_validation(
        by_class, cg, want, rng, strict=strict_targets)

    # ---- assemble ------------------------------------------------------------
    cls_idx = {c: i for i, c in enumerate(classes)}
    rel = lambda p: os.path.relpath(p, root).replace("\\", "/")

    def mk(role: str, sel: Dict[str, List[Tuple[str, int]]],
           src_split: str, recipe: Dict[str, Any]) -> SplitManifest:
        files, labels, rows, srcs, groups = [], [], [], [], []
        for c in classes:                       # class-major order, as the cache is
            for p, i in sel[c]:
                files.append(rel(p))
                labels.append(cls_idx[c])
                rows.append(i)
                srcs.append(src_split)
                groups.append(group_key(p, "image"))
        return SplitManifest(bed=bed, role=role, root=root, class_names=list(classes),
                             files=files, labels=labels, source_bed=bed,
                             source_splits=srcs, source_rows=rows,
                             group_unit="image", groups=groups, recipe=recipe)

    test_sel: Dict[str, List[Tuple[str, int]]] = {c: [] for c in classes}
    for p, c, i in te:
        test_sel[c].append((p, i))

    base = dict(seed=int(seed), content_group=content_group,
                valid_frac=float(valid_frac),
                source_corpus=os.path.basename(os.path.abspath(root)),
                published_split=True)

    bm = BedManifest(bed=bed, root=root, class_names=list(classes),
                     group_unit="image", seed=int(seed),
                     built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    bm.splits["train"] = mk("train", train_pool, tr_folder,
                            dict(base, rule=f"published {tr_folder} folder minus the "
                                            f"validation carve-out"))
    bm.splits["valid"] = mk("valid", valid_sel, tr_folder,
                            dict(base, rule=f"{valid_frac:.0%} of each class carved out of "
                                            f"{tr_folder}, content groups kept intact",
                                 per_class_target=dict(want)))
    bm.splits["test"] = mk("test", test_sel, te_folder,
                           dict(base, rule=f"published {te_folder} folder, adopted verbatim; "
                                           f"not ours to choose"))

    # ---- audits --------------------------------------------------------------
    bm.audit["group_overlap"] = assert_disjoint(bm.splits)
    bm.audit["content_groups_per_class"] = group_audit
    bm.audit["content_group_mode"] = content_group
    bm.audit["file_overlap"] = {
        f"{a}_{b}": len(set(bm.splits[a].files) & set(bm.splits[b].files))
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    bm.audit["provenance"] = prov
    bm.audit["targets"] = {role: bm.splits[role].class_counts() for role in ROLES}
    bm.audit["target_match"] = {"test": prov["matches_published"]["test"],
                                "train_plus_valid": True}
    recon = {c: bm.splits["train"].class_counts()[c] + bm.splits["valid"].class_counts()[c]
             for c in classes}
    bm.audit["train_valid_reconstructs_published_train"] = (
        recon == {c: int(got["train"][c]) for c in classes})
    bm.audit["class_balanced_test"] = (
        len(set(bm.splits["test"].class_counts().values())) == 1)
    bm.audit["notes"] = [
        "the evaluation cohort is the corpus's own published folder, adopted verbatim",
        "validation is carved out of the training folder only; no valid/ folder exists",
        (f"validation groups are connected components of {DEDUP_RULE_ID}, so no "
         f"validation row is a strict duplicate of a training row"
         if content_group == "strict" else
         f"validation groups are '{content_group}' content keys, which do not see "
         f"reflected or re-encoded copies; content_group='strict' is the mode that "
         f"keeps duplicate pairs off both sides of the train/valid cut"),
        ("the test cohort is NOT class-balanced, so macro recall is not accuracy and "
         "macro FNR is not 100 - accuracy" if not bm.audit["class_balanced_test"] else
         "the test cohort is class-balanced, so macro recall equals accuracy"),
        (f"{prov['augmented_filenames']['test']} test file(s) are named as "
         f"augmentations by the archive itself and are kept, because the cohort is "
         f"adopted verbatim; see bed 'nick_dedup' for the leakage-controlled variant"
         if prov["augmented_filenames"]["test"] else
         "no test file is named as an augmentation"),
    ]
    if not bm.audit["train_valid_reconstructs_published_train"]:
        raise RuntimeError(
            f"bed {bed}: train + valid = {recon} does not reconstruct the {tr_folder} "
            f"folder { {c: int(got['train'][c]) for c in classes} }; the carve lost rows")
    return bm

# ---------------------------------------------------------------------------
# Builder: the same published corpus, with train->test leakage removed
# ---------------------------------------------------------------------------
# Rejection ground (2) requires the final test cohort to "stay fully independent
# of method development". Adopting the archive's published split satisfies that
# for *our* choices -- we did not pick the cohort -- but it does not make the
# cohort independent of the training data, and in this archive it is not: about
# 70% of the real test files are the same picture as a real training file, usually
# mirrored. Measured with the criterion below, on files whose names do not contain
# "aug": glioma 117/400, meningioma 270/297, notumor 360/400, pituitary 295/400.
# Confirmed at 128x128, 256x256 and native resolution; 197 of 200 sampled
# meningioma pairs have identical native dimensions, brightness-matched residuals
# average 0.65 grey levels, and 25 are identical pixel for pixel.
#
# That overlap is the archive's, not ours. Inside a verified pair the two halves
# carry JPEG quantization tables from two different pre-existing families, in
# opposite order, and in pituitary not one of the 295 pairs shares a table; a
# local flip-and-save would have stamped a single new table on every copy it
# wrote. Duplicate rate is also flat across the test file numbering. So the
# duplication came with the merge of figshare + SARTAJ + Br35H and no
# re-extraction removes it.
#
# Hence two beds over one corpus, reported side by side:
#
#   nick        the published split verbatim -- comparable to prior work, and
#               carrying prior work's leakage along with it
#   nick_dedup  the same test cohort, byte for byte, with every training row that
#               duplicates a test row removed
#
# Three properties make this defensible rather than convenient:
#
#   * REMOVAL IS FROM TRAIN ONLY. The test cohort of nick_dedup is the same file
#     list, in the same order, as the test cohort of nick -- assert_test_cohorts_agree
#     checks it. So the two numbers are measured on one cohort and differ only in
#     what was learned from, which is the only way the drop between them means
#     anything.
#   * THE RULE IS LABEL-FREE AND SCORE-FREE. The mask is a deterministic function
#     of pixels: no label, no prediction, no metric enters it. It cannot be tuned
#     toward a better result, and its only possible effect on the reported number
#     is to lower it by shrinking the training set. It is written to
#     splits_v5/dedup_mask_nick.json before anything is fitted, so the removal
#     list is a reviewable artifact rather than a claim.
#   * THE RULE IS FIXED IN ADVANCE, HERE, as five named constants. It is not swept.
#
# Published precedent for reporting both: MDPI Computers 15(9):586 evaluates "on
# the standard and image-level deduplicated splits of the Nickparvar brain tumour
# MRI dataset under a three-seed, leakage-aware protocol" and reports 99.42+-0.16%
# standard against 95.25+-0.32% deduplicated -- a 4.2-point inflation that five
# ImageNet-pretrained baselines reproduce.

# The rule, fixed a priori. Two stages because one is not enough: a single 32x32
# threshold either admits neighbouring slices of the same patient (which are not
# copies) or misses re-encoded copies (which are). Stage 1 is a cheap recall filter
# over all 8 dihedral transforms; stage 2 is the decision, and it must pass BOTH a
# correlation floor and an absolute photometric test, because correlation
# saturates on low-detail images -- notumor is exactly that case, which is why its
# pairs are the softest of the four classes.
DEDUP_SCREEN_SIDE = 32          # stage 1 grid
DEDUP_SCREEN_THRESH = 0.995     # stage 1: keep as candidate (recall, not decision)
DEDUP_VERIFY_SIDE = 128         # stage 2 grid
DEDUP_VERIFY_THRESH = 0.9995    # stage 2: correlation floor
DEDUP_MAX_RESIDUAL = 3.0        # stage 2: mean |residual| in grey levels, after
                                # least-squares brightness/contrast matching
DEDUP_RULE_ID = "dihedral8-32@0.995->128@0.9995+resid<=3.0"


def _dedup_grid(path: str, side: int) -> np.ndarray:
    """side x side float32 grayscale grid, 2-D so np.rot90 applies."""
    from PIL import Image                                    # local import on purpose

    with Image.open(path) as im:
        g = im.convert("L").resize((side, side), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32)


def _dihedral_views(a: np.ndarray) -> List[np.ndarray]:
    """The eight dihedral transforms of a 2-D grid, each C-contiguous."""
    out: List[np.ndarray] = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(np.ascontiguousarray(r))
        out.append(np.ascontiguousarray(np.fliplr(r)))
    return out


def _unit(a: np.ndarray) -> np.ndarray:
    """Mean-removed, unit-norm flattening. Zero for a constant grid, never NaN."""
    v = a.reshape(-1) - float(a.mean())
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def brightness_matched_residual(x: np.ndarray, b: np.ndarray) -> float:
    """Mean |a*x + c - b| under the least-squares (a, c) that best maps x onto b.

    Correlation is blind to brightness and contrast, which is what makes it the
    right screen and the wrong decision: two different slices windowed the same way
    can correlate at 0.999 on a coarse grid. This puts the disagreement back in grey
    levels, so "the same picture, re-encoded" (a fraction of a level) separates from
    "a similar picture" (several levels) on an absolute scale a reader can judge.
    """
    xc = x.reshape(-1) - float(x.mean())
    bc = b.reshape(-1) - float(b.mean())
    den = float(xc @ xc)
    a = (float(xc @ bc) / den) if den > 0.0 else 0.0
    return float(np.abs(a * xc - bc).mean())


def same_picture(a_grid: np.ndarray, b_grid: np.ndarray) -> Tuple[bool, float, int, float]:
    """(verdict, best_corr, dihedral_index, residual) at stage-2 resolution."""
    views = _dihedral_views(a_grid)
    bz = _unit(b_grid)
    sc = np.stack([_unit(v) for v in views]) @ bz
    k = int(np.argmax(sc))
    r = brightness_matched_residual(views[k], b_grid)
    ok = bool(sc[k] >= DEDUP_VERIFY_THRESH and r <= DEDUP_MAX_RESIDUAL)
    return ok, float(sc[k]), k, r


# Names for the index same_picture returns, in the order _dihedral_views emits them:
# rotate by k, then mirror that. Deliberately NOT called DIHEDRAL_NAMES. The audit
# half of this file has a constant by that name whose order is the one
# _dihedral_perms emits (all four rotations first, then all four mirrors), and the
# two tuples disagree in seven of their eight positions. Both were module-level
# DIHEDRAL_NAMES until 2026-09-05; because Python resolves globals at call time, the
# later definition won for both readers, so every pair in a freshly built dedup mask
# would have carried the wrong transform name while still being the right pair. Two
# orderings, two names, no shadowing.
DEDUP_DIHEDRAL_NAMES = ("identity", "flip", "rot90", "rot90+flip",
                        "rot180", "rot180+flip", "rot270", "rot270+flip")


def leaked_train_rows(train_paths: Sequence[str], test_paths: Sequence[str],
                      labels_train: Optional[Sequence[int]] = None,
                      labels_test: Optional[Sequence[int]] = None,
                      top_k: int = 40, verbose: bool = True) -> Dict[str, Any]:
    """Which training files are the same picture as some test file.

    Returns {"leaked": [train indices], "pairs": [...], "rule": ..., "counts": ...}.

    Direction matters and is deliberate: this asks "which TRAIN row must go", never
    "which test row must go". The test cohort is the published one and is not ours
    to edit.

    Cost. The exact answer is |train| x |test| x 8 comparisons at 128x128, which is
    ~9e6 pairs here and hours of work. The screen makes it tractable without
    changing the verdict: stage 1 runs the same 8 transforms at 32x32 as one matrix
    product, and only candidates at or above DEDUP_SCREEN_THRESH are decided at
    128x128. Since a pair that scores below 0.995 on a coarse grid cannot reach
    0.9995 on a fine one -- downsampling averages away disagreement, it cannot
    create it -- the screen costs recall only in the direction of keeping MORE
    training rows, i.e. against us. `top_k` bounds the per-file verification work;
    it is the number of candidates examined, not a cap on removals.
    """
    tr, te = list(train_paths), list(test_paths)
    if not tr or not te:
        raise ValueError("both folders must be non-empty to compute a leakage mask")

    if verbose:
        print(f"  dedup: decoding {len(te)} test + {len(tr)} train images at "
              f"{DEDUP_SCREEN_SIDE}x{DEDUP_SCREEN_SIDE} ...", flush=True)
    TE = np.stack([_unit(_dedup_grid(p, DEDUP_SCREEN_SIDE)) for p in te]).astype(np.float32)

    leaked: Dict[int, Tuple[int, float, int, float]] = {}
    m = DEDUP_SCREEN_SIDE * DEDUP_SCREEN_SIDE
    block = 256
    for s in range(0, len(tr), block):
        chunk = tr[s:s + block]
        grids = [_dedup_grid(p, DEDUP_SCREEN_SIDE) for p in chunk]
        Q = np.stack([np.stack([_unit(v) for v in _dihedral_views(g)])
                      for g in grids]).astype(np.float32)
        S = (Q.reshape(-1, m) @ TE.T).reshape(len(chunk), 8, len(te)).max(axis=1)
        for bi in range(len(chunk)):
            i = s + bi
            order = np.argsort(-S[bi])[:int(top_k)]
            cands = [int(j) for j in order if S[bi][j] >= DEDUP_SCREEN_THRESH]
            if not cands:
                continue
            a = _dedup_grid(tr[i], DEDUP_VERIFY_SIDE)
            for j in cands:
                ok, corr, k, r = same_picture(a, _dedup_grid(te[j], DEDUP_VERIFY_SIDE))
                if ok:
                    leaked[i] = (j, corr, k, r)
                    break
        if verbose and (s // block) % 4 == 0:
            print(f"    screened {min(s + block, len(tr))}/{len(tr)} train files, "
                  f"{len(leaked)} leaked so far", flush=True)

    pairs = [{"train": tr[i], "test": te[j], "corr": round(c, 6),
              "transform": DEDUP_DIHEDRAL_NAMES[k], "residual": round(r, 4)}
             for i, (j, c, k, r) in sorted(leaked.items())]
    out: Dict[str, Any] = {
        "leaked": sorted(leaked),
        "pairs": pairs,
        "rule": DEDUP_RULE_ID,
        "params": dict(screen_side=DEDUP_SCREEN_SIDE, screen_thresh=DEDUP_SCREEN_THRESH,
                       verify_side=DEDUP_VERIFY_SIDE, verify_thresh=DEDUP_VERIFY_THRESH,
                       max_residual=DEDUP_MAX_RESIDUAL, top_k=int(top_k)),
        "counts": {"train": len(tr), "test": len(te), "leaked_train": len(leaked),
                   "distinct_test_partners": len({j for j, _, _, _ in leaked.values()})},
    }
    # A pair whose two halves carry different labels is not leakage, it is label
    # noise, and it caps achievable accuracy no matter what the method does. Worth
    # counting separately rather than folding into one number.
    if labels_train is not None and labels_test is not None:
        out["counts"]["cross_label_pairs"] = sum(
            1 for i, (j, _, _, _) in leaked.items()
            if int(labels_train[i]) != int(labels_test[j]))
    return out


DEDUP_MASK_NAME = "dedup_mask_{bed}.json"
STRICT_PAIRS_NAME = "strict_pairs_{corpus}.json"
LEAKAGE_AUDIT_NAME = "leakage_audit_{bed}.json"


def _corpus_tag(root: str) -> str:
    """'.../balance 4class' -> 'balance-4class'. Filename-safe corpus identity.

    Keyed on the folder rather than on the bed on purpose: the strict pair set is a
    property of the pixels, and four beds (bal, imb, nick, nick_dedup) enumerate one
    folder. Sharing the cache between them is what makes auditing the second bed
    almost free, and the digest check below is what makes the sharing safe.
    """
    base = os.path.basename(os.path.abspath(str(root))).lower()
    return re.sub(r"[^a-z0-9]+", "-", base).strip("-") or "corpus"


def _digest_list(items: Sequence[str]) -> str:
    """Order-sensitive identity of a file list. Same convention as SplitManifest.digest."""
    h = hashlib.md5()
    for s in items:
        h.update(str(s).replace("\\", "/").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def dedup_mask_path(bed: str = "nick", out_dir: Optional[str] = None) -> str:
    return os.path.join(out_dir or SPLIT_DIR, DEDUP_MASK_NAME.format(bed=bed))


def strict_pairs_path(root: str, out_dir: Optional[str] = None) -> str:
    return os.path.join(out_dir or SPLIT_DIR,
                        STRICT_PAIRS_NAME.format(corpus=_corpus_tag(root)))


def leakage_audit_path(bed: str, out_dir: Optional[str] = None) -> str:
    return os.path.join(out_dir or SPLIT_DIR, LEAKAGE_AUDIT_NAME.format(bed=bed))


# ---------------------------------------------------------------------------
# The strict relation over a whole corpus: one scan, three consumers
# ---------------------------------------------------------------------------
# leaked_train_rows answers one direction of one question -- which TRAIN row
# duplicates a TEST row -- and bounds its own work with top_k. Two other questions
# need the same relation without that direction and without that bound:
#
#   * the GATE (strict_leakage_audit). Does any pair cross a role boundary, under
#     the rule that decides REMOVAL rather than under a looser correlation
#     threshold? Computed with no per-file candidate bound, so it is an exact CHECK
#     of the mask's top_k rather than a second run of the same heuristic: if
#     top_k=40 ever missed a leaked row, bed nick_dedup fails its own gate instead
#     of quietly shipping a contaminated number.
#   * the CARVE (content_group='strict'). Validation must not hold a strict
#     duplicate of a training row. That is a symmetric, connected-component
#     property of the training pool alone, which no per-file key function can
#     express -- A~B and B~C have to put A and C in one group.
#
# All of it reads one cached artifact, keyed on the corpus FOLDER and not on the
# bed, because the relation is a property of the pixels: bal, imb, nick and
# nick_dedup all enumerate `balance 4class`, so the corpus is scanned once.

def strict_duplicate_pairs(root: str, class_names: Optional[Sequence[str]] = None,
                           folders: Sequence[str] = ("Train", "Test"),
                           cache_path: Optional[str] = None,
                           reuse: bool = True, write: bool = True,
                           verbose: bool = True, block: int = 512) -> Dict[str, Any]:
    """Every pair of images under `root` that DEDUP_RULE_ID calls the same picture.

    Returns {"files": [relpath, ...], "pairs": [[i, j, corr, t, residual], ...]}
    plus provenance. i < j index into "files"; t indexes DEDUP_DIHEDRAL_NAMES.

    The same two stages the removal rule uses, through the same same_picture(), so
    the gate and the mask cannot drift apart: a 32x32 correlation screen at
    DEDUP_SCREEN_THRESH over all pairs, then a 128x128 decision requiring both
    DEDUP_VERIFY_THRESH and a brightness-matched residual within
    DEDUP_MAX_RESIDUAL. The screen cannot lose a true pair -- downsampling averages
    disagreement away, it cannot create it -- so this is the exact relation under
    the rule rather than an approximation of it.

    Cost, measured on the 7,200-image four-class corpus: stage 1 is the same
    all-pairs scan step 6 already ran for the 0.98 audit; stage 2 decodes the ~5.4k
    files that appear in some candidate pair (about a minute) and verifies ~14k
    candidates in seconds. CPU only, pixels only -- no labels, no features, no
    model output, so nothing here is a test read in the SelectionLedger sense.
    """
    cls = list(class_names or FOUR_CLASSES)
    paths: List[str] = []
    rels: List[str] = []
    for sp in folders:
        for c in cls:
            for p in list_images(os.path.join(root, sp, c)):
                paths.append(p)
                rels.append(os.path.relpath(p, root).replace("\\", "/"))
    if not paths:
        raise RuntimeError(f"no images under {root} for folders {tuple(folders)}")

    want = {"rule": DEDUP_RULE_ID, "folders": [str(f) for f in folders],
            "classes": cls, "files_digest": _digest_list(rels)}
    cp = cache_path or strict_pairs_path(root)
    if reuse and os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if all(cached.get(k) == v for k, v in want.items()):
                if verbose:
                    print(f"  strict: reusing {os.path.basename(cp)} "
                          f"({len(cached.get('pairs', []))} pair(s) over "
                          f"{len(cached.get('files', []))} file(s))", flush=True)
                return cached
            if verbose:
                print(f"  strict: {os.path.basename(cp)} was built for a different rule, "
                      f"folder set or file list; recomputing", flush=True)
        except (OSError, ValueError) as e:
            print(f"  strict: could not read {cp} ({e}); recomputing", flush=True)
    return _strict_pairs_compute(root, paths, rels, want, cp, write, verbose, block)


def _strict_pairs_compute(root: str, paths: List[str], rels: List[str],
                          want: Dict[str, Any], cp: str, write: bool,
                          verbose: bool, block: int) -> Dict[str, Any]:
    """Stage 1 over all pairs, stage 2 over the survivors. See strict_duplicate_pairs."""
    n = len(paths)
    if verbose:
        print(f"  strict: decoding {n} images at {DEDUP_SCREEN_SIDE}x{DEDUP_SCREEN_SIDE} "
              f"for the screen ...", flush=True)
    t0 = time.time()
    Z = _znorm(_load_descriptors(paths, side=DEDUP_SCREEN_SIDE))
    if verbose:
        print(f"  strict: screening all pairs at >= {DEDUP_SCREEN_THRESH} ...", flush=True)
    cand = near_duplicate_pairs(Z, side=DEDUP_SCREEN_SIDE,
                                thresh=DEDUP_SCREEN_THRESH, block=block)
    del Z
    if verbose:
        print(f"  strict: {len(cand)} candidate pair(s) in {time.time() - t0:.0f}s; "
              f"deciding at {DEDUP_VERIFY_SIDE}x{DEDUP_VERIFY_SIDE}", flush=True)

    # Only the files that appear in some candidate pair are decoded at stage-2
    # resolution, and each is decoded once. Holding all 7,200 at 128x128 would be
    # 470 MB; holding the ~5.4k that matter is 350 MB, and the alternative --
    # decoding per pair -- is ~28k decodes for ~5.4k files.
    need = sorted({i for i, _, _, _ in cand} | {j for _, j, _, _ in cand})
    pos = {g: k for k, g in enumerate(need)}
    G = np.empty((len(need), DEDUP_VERIFY_SIDE, DEDUP_VERIFY_SIDE), dtype=np.float32)
    for k, g in enumerate(need):
        G[k] = _dedup_grid(paths[g], DEDUP_VERIFY_SIDE)
        if verbose and (k + 1) % 1500 == 0:
            print(f"    decoded {k + 1}/{len(need)}", flush=True)

    pairs: List[List[Any]] = []
    for i, j, _s32, _t32 in cand:
        ok, corr, t, resid = same_picture(G[pos[i]], G[pos[j]])
        if ok:
            pairs.append([int(i), int(j), round(float(corr), 6), int(t),
                          round(float(resid), 4)])
    del G

    out = dict(want)
    out.update(
        mode="strict_duplicate_relation",
        root=os.path.basename(os.path.abspath(root)),
        params=dict(screen_side=DEDUP_SCREEN_SIDE, screen_thresh=DEDUP_SCREEN_THRESH,
                    verify_side=DEDUP_VERIFY_SIDE, verify_thresh=DEDUP_VERIFY_THRESH,
                    max_residual=DEDUP_MAX_RESIDUAL, top_k=None),
        files=rels, pairs=pairs,
        counts=dict(n_images=n, n_candidates=len(cand), n_pairs=len(pairs),
                    n_files_verified=len(need)),
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        caveat=("Exhaustive under DEDUP_RULE_ID: every pair was screened, so unlike "
                "the dedup mask this is not bounded by a per-file top_k. Still a lower "
                "bound on identity in general -- the rule is invariant to reflection, "
                "to rotation by multiples of 90 degrees and to affine intensity change, "
                "and to nothing else."))
    if write:
        os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(out, fh)              # no indent: ~5.6k pairs, machine-read only
        if verbose:
            print(f"  strict: {len(pairs)} verified pair(s); wrote {cp}", flush=True)
    return out


def strict_duplicate_relation(pairs_report: Dict[str, Any],
                              root: Optional[str] = None) -> Dict[str, List[str]]:
    """{path -> [its strict duplicate partners]}, ready for _content_groups('strict').

    Absolute paths when `root` is given, corpus-relative when it is not. The carve
    keys groups by absolute path because by_class holds absolute paths, so pass root.
    """
    files = pairs_report["files"]
    if root is not None:
        files = [os.path.join(root, f.replace("/", os.sep)) for f in files]
    rel: Dict[str, List[str]] = collections.defaultdict(list)
    for i, j, *_ in pairs_report["pairs"]:
        rel[files[i]].append(files[j])
        rel[files[j]].append(files[i])
    return dict(rel)


# Beds excused from the strict gate, each with the reason that has to be printed
# instead. Exemption is not leniency: it is the difference between a bed whose
# disjointness is a claim of ours and a bed whose overlap is the finding.
#
# `nick` is the published split adopted verbatim. Its train-test overlap came with
# the archive; it is the quantity the nick / nick_dedup pair exists to measure, and
# it is reported in the manuscript as such. Gating it would stop every run at step 6
# to re-announce a measurement the paper already makes, and the only way to make the
# gate pass would be to stop adopting the published split -- which is the same-cohort
# comparison rejection ground (1) asks for.
#
# `bal` and `imb` are non-reportable (see NON_REPORTABLE_BEDS) and ~72% Test-to-Train
# duplicated. Nothing they measure may be quoted, so there is nothing to protect.
#
# Not exempt, deliberately: nick_dedup and both pw beds. nick_dedup is where the
# leakage claim actually lives -- the mask removes every strict train-test pair by
# construction, so its gate is expected to pass at zero and a failure means the mask
# and the rule have drifted apart, which is exactly the silent failure worth
# catching. The pw beds are patient-grouped by our own rule, so any cross-role
# duplicate there is our defect and has to stop the run.
#
# Lookup is by EXACT bed key, never through _bed_family(). That helper maps
# nick_dedup -> nick, because for cache purposes they are one extraction; using it
# here would inherit nick's exemption and silently switch off the only gate that is
# expected to fire, on the only bed whose cleanliness is a claim of ours. (It did,
# for one revision: caught 2026-09-05 by a behavioural check asserting
# gate_exempt['nick_dedup'] is False.) A bed is exempt only if its own name is a key
# of this dict.
GATE_EXEMPT_BEDS: Dict[str, str] = {
    "nick": ("the published four-class split, adopted verbatim. Its train-test overlap "
             "is the archive's and is the quantity bed 'nick_dedup' exists to measure, "
             "so it is disclosed rather than gated; removing it here would forfeit the "
             "same-cohort comparison the bed is for"),
    "bal": ("non-reportable re-cut of the four-class corpus (see NON_REPORTABLE_BEDS); "
            "its ~72% Test-to-Train duplication is a known property of a retired bed, "
            "not a defect a gate can prevent"),
    "imb": ("non-reportable invented subsample (see NON_REPORTABLE_BEDS); shares the "
            "balanced bed's validation and test rows and its duplication with them"),
}


def strict_leakage_audit(root: str, manifests: Dict[str, SplitManifest],
                         class_names: Optional[Sequence[str]] = None,
                         folders: Sequence[str] = ("Train", "Test"),
                         pairs_report: Optional[Dict[str, Any]] = None,
                         out_dir: Optional[str] = None,
                         bed: str = "nick", max_examples: int = 300,
                         reuse: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Cross-role duplicate counts under DEDUP_RULE_ID -- the rule that removes rows.

    This is the gate's measurement, and it is deliberately a different question from
    orientation_duplicate_audit's. That one counts pairs at a 0.98 correlation on a
    32x32 grid: a description of the corpus, loose enough to catch different slices
    of one patient, and therefore not a criterion anything can be required to pass.
    This one counts pairs under the exact two-stage rule that decides which training
    rows bed nick_dedup may not see. A gate whose criterion is stricter than its
    remedy can never be satisfied by applying the remedy, which is the whole reason
    for the change: nick_dedup passes this at zero BY CONSTRUCTION, so the gate is
    non-vacuous rather than unsatisfiable.

    Three counts, reported separately because they have different consequences:

      train_test, valid_test   contaminate the reported number. Their SUM is what
                               the gate fires on -- the test cohort is what must
                               stay independent of everything upstream of it.
      train_valid              inflates the validation score that picked the
                               configuration, the weights and the thresholds. Not
                               gated, because it is a defect in the carve rather
                               than in the test cohort, and content_group='strict'
                               is its fix; reported so it cannot be ignored.

    Cross-class pairs are tallied apart from all three. A duplicate pair whose halves
    carry different labels is label noise, not leakage: it caps achievable accuracy
    and no split can remove it. Folding it into a leakage count would overstate the
    leakage and hide the ceiling.
    """
    rep = pairs_report or strict_duplicate_pairs(
        root, class_names=class_names, folders=folders,
        cache_path=strict_pairs_path(root, out_dir), reuse=reuse, verbose=verbose)
    files = rep["files"]
    role_of: Dict[str, str] = {}
    class_of: Dict[str, str] = {}
    for role, m in manifests.items():
        for k, f in enumerate(m.files):
            key = f.replace("\\", "/")
            role_of[key] = role
            class_of[key] = m.class_names[int(m.labels[k])]

    role_same: Dict[str, int] = collections.Counter()
    role_cross_class: Dict[str, int] = collections.Counter()
    examples: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    rows_touched: Dict[str, set] = collections.defaultdict(set)
    for i, j, corr, t, resid in rep["pairs"]:
        a, b = files[i], files[j]
        ra, rb = role_of.get(a), role_of.get(b)
        if not ra or not rb:
            continue                      # a file no manifest claims: not this bed's row
        key = "_".join(sorted((ra, rb)))
        ca, cb = class_of.get(a), class_of.get(b)
        (role_cross_class if ca != cb else role_same)[key] += 1
        if ra != rb:
            # Which rows a fix would have to touch, counted on the side that could
            # legitimately move: never the test cohort, which is not ours to edit.
            rows_touched[key].add(b if ra == "test" else a)
            if len(examples[key]) < max_examples:
                examples[key].append(dict(a=a, b=b, a_role=ra, b_role=rb,
                                          a_class=ca, b_class=cb,
                                          corr=corr, transform=DEDUP_DIHEDRAL_NAMES[t],
                                          residual=resid))

    tf_same = int(role_same.get("test_train", 0) + role_same.get("test_valid", 0))
    tf_cross = int(role_cross_class.get("test_train", 0)
                   + role_cross_class.get("test_valid", 0))
    covered = sum(1 for f in files if f in role_of)
    report: Dict[str, Any] = dict(
        mode="strict_leakage_audit", bed=bed, rule=DEDUP_RULE_ID,
        params=dict(rep["params"]),
        root=os.path.basename(os.path.abspath(root)),
        n_images=len(files), n_strict_pairs=len(rep["pairs"]),
        manifest_coverage=covered,
        split_sizes={r: m.n for r, m in sorted(manifests.items())},
        role_pair_counts=dict(sorted(role_same.items())),
        role_pair_counts_cross_class=dict(sorted(role_cross_class.items())),
        n_train_test=int(role_same.get("test_train", 0)),
        n_valid_test=int(role_same.get("test_valid", 0)),
        n_train_valid=int(role_same.get("train_valid", 0)),
        n_test_facing=tf_same,
        n_test_facing_cross_class=tf_cross,
        n_cross_class_pairs=int(sum(role_cross_class.values())),
        rows_touched={k: len(v) for k, v in sorted(rows_touched.items())},
        examples={k: v for k, v in sorted(examples.items())},
        gate_exempt=bool(str(bed).strip().lower() in GATE_EXEMPT_BEDS),
        gate_exempt_reason=GATE_EXEMPT_BEDS.get(str(bed).strip().lower(), ""),
        caveat=("Counted under the removal rule itself, not under the 0.98 correlation "
                "audit, so 'gate passed' means 'no pair the deduplicator would have "
                "removed crosses a role boundary'. Cross-class pairs are label noise "
                "and are tallied separately; they are not leakage and no split removes "
                "them. Independence beyond this rule is not claimed: it is invariant "
                "to reflection, to rotation by multiples of 90 degrees and to affine "
                "intensity change, and to nothing else."),
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not covered:
        report["warning"] = ("no manifest file matched the corpus enumeration; the "
                             "role-level counts are empty and mean nothing")
    return report


def build_deduplicated_four_class_bed(
    root: str,
    bed: str = "nick_dedup",
    source_bed_manifest: Optional[BedManifest] = None,
    seed: int = 2026,
    valid_frac: float = 0.15,
    content_group: str = "md5",
    published: Optional[Dict[str, Dict[str, int]]] = None,
    mask: Optional[Dict[str, Any]] = None,
    mask_path: Optional[str] = None,
    write_mask: bool = True,
    reuse_mask: bool = True,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> BedManifest:
    """`nick` with every training row that duplicates a test row removed.

    The test cohort is copied from the `nick` manifest without recomputation, so it
    is the same files in the same order -- that identity is what makes the pair of
    numbers a controlled contrast rather than two unrelated measurements, and
    assert_test_cohorts_agree asserts it.

    Removal happens BEFORE validation is carved out, so a leaked row cannot survive
    by landing in validation, and the carve then runs on the surviving pool with the
    same rule and seed as `nick`. Validation is therefore not the same rows as
    `nick`'s validation; it cannot be, since the pool it comes from is smaller. Only
    the test cohort is shared, and only the test cohort is reported.

    The mask is cached at splits_v5/dedup_mask_nick.json keyed by rule id and by the
    digest of both file lists, and reused when all three match. Pass `mask` to
    supply one, or delete the file to force recomputation. Recomputing is a
    CPU-only image pass -- no GPU, no cache, no test read in the SelectionLedger
    sense, since the pixels are being hashed rather than the model evaluated.

    content_group is forwarded to both the source bed and the re-carve. Under
    'strict' the surviving pool is grouped by connected components of the same
    relation the mask came from, so the carve cannot put one member of a duplicate
    group in train and another in validation: measured here, 372 such pairs under
    md5 grouping against 0 under 'strict', at the same 3661/647 sizes.
    """
    src = source_bed_manifest or build_published_four_class_bed(
        root, bed="nick", seed=seed, valid_frac=valid_frac,
        content_group=content_group, published=published,
        out_dir=out_dir, verbose=verbose)

    classes = list(src.class_names)
    # The published training folder is the union of nick's train and valid rows:
    # nick carved validation out of it, and this bed has to start from the whole
    # folder again so that a leaked row cannot hide in nick's validation slice.
    #
    # `source_rows` is carried through untouched, not recomputed. It is the row
    # index into the Train cache, and it is the reason this bed costs no GPU time:
    # dropping and re-carving rows is fancy-indexing into features that already
    # exist. Re-deriving it from a fresh enumeration would work today and break the
    # first time a file is added to the folder.
    pool: List[Tuple[str, int, int, str, str]] = []      # (rel, label, row, folder, group)
    for role in ("train", "valid"):
        sm = src.splits[role]
        for k in range(sm.n):
            pool.append((sm.files[k], int(sm.labels[k]), int(sm.source_rows[k]),
                         sm.source_splits[k], sm.groups[k]))
    pool.sort(key=lambda t: t[0])
    tr_rel = [t[0] for t in pool]
    tr_lab = [t[1] for t in pool]

    te_sm = src.splits["test"]
    abspath = lambda rel: os.path.join(root, rel.replace("/", os.sep))
    tr_abs = [abspath(r) for r in tr_rel]
    te_abs = [abspath(r) for r in te_sm.files]

    want_key = {"rule": DEDUP_RULE_ID,
                "train_digest": _digest_list(tr_rel),
                "test_digest": _digest_list(list(te_sm.files))}
    mp = mask_path or dedup_mask_path("nick")
    if mask is None and reuse_mask and os.path.exists(mp):
        try:
            with open(mp, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if all(cached.get(k) == v for k, v in want_key.items()):
                mask = cached
                if verbose:
                    print(f"  dedup: reusing mask {os.path.basename(mp)} "
                          f"({len(cached.get('leaked_files', []))} training rows)",
                          flush=True)
            elif verbose:
                print(f"  dedup: mask at {os.path.basename(mp)} was built for a "
                      f"different rule or file list; recomputing", flush=True)
        except (OSError, ValueError) as e:
            print(f"  dedup: could not read {mp} ({e}); recomputing", flush=True)

    if mask is None:
        res = leaked_train_rows(tr_abs, te_abs, labels_train=tr_lab,
                               labels_test=te_sm.labels, verbose=verbose)
        mask = dict(want_key)
        mask["leaked_files"] = [tr_rel[i] for i in res["leaked"]]
        mask["counts"] = res["counts"]
        mask["params"] = res["params"]
        mask["pairs"] = [dict(p,
                              train=os.path.relpath(p["train"], root).replace("\\", "/"),
                              test=os.path.relpath(p["test"], root).replace("\\", "/"))
                         for p in res["pairs"]]
        mask["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if write_mask:
            os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
            with open(mp, "w", encoding="utf-8") as fh:
                json.dump(mask, fh, indent=2)
            if verbose:
                print(f"  dedup: wrote {mp}", flush=True)

    drop = set(mask.get("leaked_files", []))
    unknown = drop - set(tr_rel)
    if unknown:
        raise RuntimeError(
            f"bed {bed}: the dedup mask names {len(unknown)} file(s) that are not in the "
            f"published training folder, e.g. {sorted(unknown)[:3]}. The mask was built "
            f"against a different corpus; delete {mp} and rebuild.")
    if not drop:
        raise RuntimeError(
            f"bed {bed}: the dedup mask is empty, so this bed would be identical to "
            f"'nick' and reporting both would imply a contrast that does not exist. "
            f"Either the corpus changed or the rule stopped matching -- check {mp}.")
    return _assemble_dedup_bed(root, bed, classes, pool, drop, src,
                               mask, seed, valid_frac, content_group, verbose,
                               out_dir=out_dir)


def _assemble_dedup_bed(root: str, bed: str, classes: List[str],
                        pool: List[Tuple[str, int, int, str, str]],
                        drop: set, src: BedManifest, mask: Dict[str, Any],
                        seed: int, valid_frac: float, content_group: str,
                        verbose: bool,
                        out_dir: Optional[str] = None) -> BedManifest:
    """Carve validation out of the surviving pool and assemble the manifest."""
    kept = [t for t in pool if t[0] not in drop]
    if not kept:
        raise RuntimeError(f"bed {bed}: deduplication removed the entire training pool")

    by_class: Dict[str, List[Tuple[str, int]]] = {c: [] for c in classes}
    meta: Dict[str, Tuple[int, int, str, str]] = {}
    for rel, lab, row, folder, grp in kept:
        by_class[classes[lab]].append((os.path.join(root, rel.replace("/", os.sep)), row))
        meta[rel] = (lab, row, folder, grp)

    thin = {c: len(v) for c, v in by_class.items() if len(v) < 4}
    if thin:
        raise RuntimeError(
            f"bed {bed}: deduplication left {thin} training images, too few to carve "
            f"validation from. The rule removed nearly everything -- check "
            f"{dedup_mask_path('nick')} before loosening it.")

    # The relation is built over the WHOLE corpus (Train + Test) and then induced on
    # the surviving pool by _union_find_groups, which is the only correct order here:
    # a train-test edge must not fuse two training groups, and the removal has
    # already dealt with every such edge. Building it over the pool alone would give
    # the same components on this corpus and would be wrong on the next one.
    #
    # The folder pair is read off `src` rather than hardcoded as ("Train", "Test").
    # source_splits records the on-disk folder of every row, so this asks the source
    # bed which folders it was built from instead of assuming the answer -- and it
    # makes the cache key of the relation match the corpus the bed actually used.
    folders = tuple(sorted({sp for role in ROLES
                            for sp in set(src.splits[role].source_splits)}))
    cg = _content_groups([p for v in by_class.values() for p, _ in v], content_group,
                         relation=_strict_relation_for(root, content_group, out_dir,
                                                       folders, classes, verbose=verbose))
    want = {c: max(2, int(round(len(by_class[c]) * float(valid_frac)))) for c in classes}
    rng = np.random.default_rng(seed)
    valid_sel, train_pool, group_audit = _carve_validation(
        by_class, cg, want, rng, strict=False)

    cls_idx = {c: i for i, c in enumerate(classes)}
    rel_of = lambda p: os.path.relpath(p, root).replace("\\", "/")

    def mk(role: str, sel: Dict[str, List[Tuple[str, int]]],
           recipe: Dict[str, Any]) -> SplitManifest:
        files, labels, rows, srcs, groups = [], [], [], [], []
        for c in classes:                       # class-major order, as the cache is
            for p, _ in sel[c]:
                r = rel_of(p)
                lab, row, folder, grp = meta[r]
                files.append(r)
                labels.append(cls_idx[c])
                rows.append(row)
                srcs.append(folder)
                groups.append(grp)
        return SplitManifest(bed=bed, role=role, root=root, class_names=list(classes),
                             files=files, labels=labels, source_bed=bed,
                             source_splits=srcs, source_rows=rows,
                             group_unit="image", groups=groups, recipe=recipe)

    base = dict(seed=int(seed), content_group=content_group,
                valid_frac=float(valid_frac),
                source_corpus=os.path.basename(os.path.abspath(root)),
                published_split=True, deduplicated=True,
                dedup_rule=mask.get("rule", DEDUP_RULE_ID),
                dedup_params=dict(mask.get("params", {})),
                dedup_removed_from="train_folder_only",
                dedup_removed=len(drop))

    bm = BedManifest(bed=bed, root=root, class_names=list(classes),
                     group_unit="image", seed=int(seed),
                     built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    bm.splits["train"] = mk("train", train_pool,
                            dict(base, rule="published Train folder, minus every row that "
                                            "duplicates a test row, minus the validation "
                                            "carve-out"))
    bm.splits["valid"] = mk("valid", valid_sel,
                            dict(base, rule=f"{valid_frac:.0%} of each class carved out of "
                                            f"the deduplicated training pool, content "
                                            f"groups kept intact",
                                 per_class_target=dict(want)))

    # The test split is COPIED, not rebuilt. Same files, same order, same labels,
    # same cache rows as `nick`. Recomputing it from the folder would give the same
    # answer today and would silently stop doing so the day the folder changes; a
    # copy cannot drift from what it is being compared against.
    te = src.splits["test"]
    bm.splits["test"] = SplitManifest(
        bed=bed, role="test", root=root, class_names=list(classes),
        files=list(te.files), labels=[int(v) for v in te.labels], source_bed="nick",
        source_splits=list(te.source_splits), source_rows=[int(v) for v in te.source_rows],
        group_unit="image", groups=list(te.groups),
        recipe=dict(base, rule="the published test folder, copied verbatim from bed "
                               "'nick'; identical cohort by construction, so the two "
                               "beds differ only in what was trained on"))

    # ---- audits --------------------------------------------------------------
    bm.audit["group_overlap"] = assert_disjoint(bm.splits)
    bm.audit["content_groups_per_class"] = group_audit
    bm.audit["content_group_mode"] = content_group
    bm.audit["file_overlap"] = {
        f"{a}_{b}": len(set(bm.splits[a].files) & set(bm.splits[b].files))
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    bm.audit["provenance"] = dict(src.audit.get("provenance", {}))
    bm.audit["targets"] = {role: bm.splits[role].class_counts() for role in ROLES}
    bm.audit["class_balanced_test"] = (
        len(set(bm.splits["test"].class_counts().values())) == 1)
    bm.audit["dedup"] = {
        "rule": mask.get("rule", DEDUP_RULE_ID),
        "params": dict(mask.get("params", {})),
        "mask_file": os.path.basename(dedup_mask_path("nick")),
        "removed_from": "train_folder_only",
        "removed": len(drop),
        "published_train_pool": len(pool),
        "surviving_train_pool": len(kept),
        "removed_fraction": round(len(drop) / max(1, len(pool)), 6),
        "counts": dict(mask.get("counts", {})),
        "per_class_removed": {c: sum(1 for r in drop
                                     if f"/{c}/" in "/" + r.replace("\\", "/"))
                              for c in classes},
    }
    bm.audit["test_identical_to_nick"] = (
        bm.splits["test"].digest() == src.splits["test"].digest())
    if not bm.audit["test_identical_to_nick"]:
        raise RuntimeError(
            f"bed {bed}: the test cohort does not match bed 'nick'. The whole point of "
            f"this bed is that the two numbers share one cohort; without that the drop "
            f"between them measures nothing.")
    bm.audit["notes"] = [
        "the test cohort is bed 'nick''s, copied verbatim -- nothing was removed from it",
        f"{len(drop)} of {len(pool)} published training rows were removed for duplicating "
        f"a test row, under a rule fixed before any fitting: {mask.get('rule')}",
        "validation is re-carved from the surviving pool, so it is NOT nick's validation; "
        "only the test cohort is shared, and only the test cohort is reported",
        (f"the carve grouped the surviving pool by connected components of "
         f"{DEDUP_RULE_ID} -- the same relation the removal used -- so no validation "
         f"row is a strict duplicate of a training row"
         if content_group == "strict" else
         f"the carve grouped by '{content_group}' content keys, which are blind to the "
         f"reflected and re-encoded copies the removal rule catches; duplicate pairs can "
         f"therefore straddle the train/valid cut here (372 of them, measured) without "
         f"touching the test cohort. content_group='strict' removes them"),
        ("the test cohort is class-balanced, so macro recall equals accuracy"
         if bm.audit["class_balanced_test"] else
         "the test cohort is NOT class-balanced, so macro FNR is not 100 - accuracy"),
    ]
    if verbose:
        d = bm.audit["dedup"]
        print(f"  dedup: kept {d['surviving_train_pool']}/{d['published_train_pool']} "
              f"training rows ({100.0 * (1 - d['removed_fraction']):.1f}%); "
              f"test cohort unchanged at {bm.splits['test'].n} rows", flush=True)
    return bm


def assert_test_cohorts_agree(a: BedManifest, b: BedManifest) -> None:
    """Raise unless two beds' test splits are the same rows in the same order.

    Called by build_all for (nick, nick_dedup). Reporting a standard and a
    deduplicated number side by side is only meaningful if the cohort underneath
    both is one cohort; this is that claim, checked rather than asserted in prose.
    """
    da, db = a.splits["test"].digest(), b.splits["test"].digest()
    if da != db:
        raise RuntimeError(
            f"beds {a.bed!r} and {b.bed!r} do not share a test cohort "
            f"({a.splits['test'].n} vs {b.splits['test'].n} rows, digests {da} vs {db}). "
            f"The pair of numbers they produce would not be comparable.")


def build_patient_bed(
    pw_root: str,
    group_unit: str = "identifier",
    seed: int = 42,
    valid_frac_groups: float = 0.2,
    reuse_existing_folders: bool = True,
    strict_targets: bool = False,
) -> BedManifest:
    """Manifest the Cheng/figshare bed at a chosen identity granularity.

    group_unit="identifier" with reuse_existing_folders=True reproduces the split
    already on disk exactly -- 1,811 / 472 / 781 slices over 141 / 34 / 58
    identifiers, matching Table I and the stored valid_split_report.json. Nothing
    moves, so the existing caches stay valid.

    group_unit="family" merges identifiers differing only by a trailing capital
    letter. Eight such families currently straddle two or three roles
    (MR029209E/G/I spans test, valid and train), so a merge *forces* a re-split:
    the whole family has to land on one side. Because every row records which
    folder and which cache index it came from, that re-split is still a
    re-indexing of features that already exist rather than a new forward pass.

    The point of supporting both is that the manuscript can then report the
    person-disjoint number next to the identifier-disjoint one instead of
    claiming they are the same thing.
    """
    if group_unit not in ("identifier", "family"):
        raise ValueError("patient bed grouping must be 'identifier' or 'family'")

    classes = discover_classes(os.path.join(pw_root, "train"))
    rows: List[Tuple[str, str, str, int, int]] = []   # path, cls, src_split, src_row, label
    cls_idx = {c: i for i, c in enumerate(classes)}
    for sp in ("train", "valid", "test"):
        for p, c, i in enumerate_split(pw_root, sp, classes):
            rows.append((p, c, sp, i, cls_idx[c]))

    keyed = [(p, c, sp, i, y, group_key(p, group_unit)) for p, c, sp, i, y in rows]
    groups: Dict[str, List[Tuple[str, str, str, int, int, str]]] = collections.defaultdict(list)
    for r in keyed:
        groups[r[5]].append(r)

    # -- decide each group's role --------------------------------------------
    straddling: Dict[str, List[str]] = {}
    assign: Dict[str, str] = {}
    if reuse_existing_folders:
        for g, items in groups.items():
            sps = sorted({it[2] for it in items})
            if len(sps) == 1:
                assign[g] = sps[0]
            else:
                # a merged family spanning roles: send it wherever the majority of
                # its slices already sit, breaking ties toward the stricter role
                straddling[g] = sps
                cnt = collections.Counter(it[2] for it in items)
                order = {"test": 0, "valid": 1, "train": 2}
                assign[g] = min(cnt, key=lambda s: (-cnt[s], order[s]))
    else:
        rng = np.random.default_rng(seed)
        gmaj = {g: collections.Counter(it[4] for it in items).most_common(1)[0][0]
                for g, items in groups.items()}
        by_lab: Dict[int, List[str]] = collections.defaultdict(list)
        for g, lab in gmaj.items():
            by_lab[lab].append(g)
        test_g: set = set()
        valid_g: set = set()
        for lab in sorted(by_lab):
            gs = sorted(by_lab[lab])
            rng.shuffle(gs)
            n_te = max(1, int(round(len(gs) * 0.25)))
            n_va = max(1, int(round(len(gs) * valid_frac_groups)))
            test_g.update(gs[:n_te])
            valid_g.update(gs[n_te:n_te + n_va])
        for g in groups:
            assign[g] = "test" if g in test_g else ("valid" if g in valid_g else "train")

    # -- build the three manifests -------------------------------------------
    rel = lambda p: os.path.relpath(p, pw_root).replace("\\", "/")
    buckets: Dict[str, List[Tuple[str, str, str, int, int, str]]] = {r: [] for r in ROLES}
    for g, items in groups.items():
        buckets[assign[g]].extend(items)

    bm = BedManifest(bed="pw", root=pw_root, class_names=list(classes),
                     group_unit=group_unit, seed=int(seed),
                     built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    for role in ROLES:
        # class-major, then by source folder and original row, so the order is a
        # deterministic function of the corpus and not of dict iteration
        items = sorted(buckets[role], key=lambda r: (cls_idx[r[1]], r[2], r[3]))
        bm.splits[role] = SplitManifest(
            bed="pw", role=role, root=pw_root, class_names=list(classes),
            files=[rel(r[0]) for r in items],
            labels=[r[4] for r in items],
            source_bed="pw",
            source_splits=[r[2] for r in items],
            source_rows=[r[3] for r in items],
            group_unit=group_unit,
            groups=[r[5] for r in items],
            recipe=dict(seed=int(seed), group_unit=group_unit,
                        reuse_existing_folders=bool(reuse_existing_folders),
                        valid_frac_groups=float(valid_frac_groups)))

    bm.audit["group_overlap"] = assert_disjoint(bm.splits)
    bm.audit["n_groups"] = {r: bm.splits[r].n_groups() for r in ROLES}
    bm.audit["n_slices"] = {r: bm.splits[r].n for r in ROLES}
    bm.audit["identifier_count"] = len({pid_from_path(p) for p, *_ in rows})
    bm.audit["family_count"] = len({pid_family(pid_from_path(p)) for p, *_ in rows})
    bm.audit["straddling_families"] = {k: v for k, v in sorted(straddling.items())}
    bm.audit["reuse_existing_folders"] = bool(reuse_existing_folders)
    bm.audit["matches_table_I"] = (bm.audit["n_slices"] == PW_TARGETS)
    if strict_targets and not bm.audit["matches_table_I"]:
        raise RuntimeError(f"patient bed slice counts {bm.audit['n_slices']} do not match "
                           f"Table I {PW_TARGETS}")
    return bm


def kfold_group_manifests(
    train_m: SplitManifest,
    valid_m: SplitManifest,
    k: int = 5,
    seed: int = 42,
) -> List[Dict[str, SplitManifest]]:
    """Group-disjoint K-fold over train+valid, leaving the test cohort alone.

    This is what the Cheng-protocol comparison needs: the published competitors
    on this corpus report cross-validated accuracy, so comparing a single
    held-out number against them is not a like-for-like comparison. Folds are
    formed over whole groups, and the test manifest is never involved, so
    cross-validation cannot leak into the sealed cohort.

    Returns [{"train": m, "valid": m}, ...] of length k, each carrying per-row
    provenance so features come from the caches that already exist.
    """
    if train_m.group_unit != valid_m.group_unit:
        raise ValueError("train and valid manifests use different grouping units")
    pool = list(zip(train_m.files + valid_m.files,
                    train_m.labels + valid_m.labels,
                    train_m.source_splits + valid_m.source_splits,
                    train_m.source_rows + valid_m.source_rows,
                    train_m.groups + valid_m.groups))

    gmaj: Dict[str, int] = {}
    gitems: Dict[str, List[Tuple[str, int, str, int, str]]] = collections.defaultdict(list)
    for row in pool:
        gitems[row[4]].append(row)
    for g, items in gitems.items():
        gmaj[g] = collections.Counter(it[1] for it in items).most_common(1)[0][0]

    # stratify fold assignment by the group's majority class, then round-robin so
    # every fold sees every class even when groups are unevenly sized
    rng = np.random.default_rng(seed)
    fold_of: Dict[str, int] = {}
    by_lab: Dict[int, List[str]] = collections.defaultdict(list)
    for g, lab in gmaj.items():
        by_lab[lab].append(g)
    for lab in sorted(by_lab):
        gs = sorted(by_lab[lab])
        rng.shuffle(gs)
        for j, g in enumerate(gs):
            fold_of[g] = j % k

    cls_idx = {c: i for i, c in enumerate(train_m.class_names)}
    out: List[Dict[str, SplitManifest]] = []
    for f in range(k):
        parts: Dict[str, List[Tuple[str, int, str, int, str]]] = {"train": [], "valid": []}
        for g, items in gitems.items():
            parts["valid" if fold_of[g] == f else "train"].extend(items)
        fold: Dict[str, SplitManifest] = {}
        for role in ("train", "valid"):
            items = sorted(parts[role], key=lambda r: (r[1], r[2], r[3]))
            fold[role] = SplitManifest(
                bed=train_m.bed, role=role, root=train_m.root,
                class_names=list(train_m.class_names),
                files=[r[0] for r in items], labels=[r[1] for r in items],
                source_bed=train_m.source_bed,
                source_splits=[r[2] for r in items],
                source_rows=[r[3] for r in items],
                group_unit=train_m.group_unit, groups=[r[4] for r in items],
                recipe=dict(cv="group_kfold", k=int(k), fold=int(f), seed=int(seed),
                            pool="train+valid", note="test cohort excluded by construction"))
        assert_disjoint(fold)
        out.append(fold)
    _ = cls_idx
    return out

# ---------------------------------------------------------------------------
# Binding manifests to the feature caches that already exist
# ---------------------------------------------------------------------------
# v4 keyed every cache file on dataset_fingerprint(), which hashes per-class file
# counts *and the latest mtime per class*. That key is dead. Recomputing it for
# all 35 (bed, split, backbone, tta) combinations reproduces none of the 54 .npy
# files on disk: the four-class corpora carry only one or two distinct mtimes per
# class from a bulk copy dated 2026-02-13, while the caches were written
# 2026-08-24, so the mtimes that went into the hash are gone. A cache key must not
# depend on filesystem metadata that a copy silently rewrites.
#
# v5 therefore binds by (backbone, img_size, split) plus an *assertion* on row
# count, and records the resolved filename and its content hash in the run
# artifact. If a bind is ambiguous it says so instead of taking the newest file
# and hoping.

@dataclass
class CacheRef:
    path: str
    rows: int
    dim: int
    split_key: str
    fmt: str            # "s{img}" (v4/v5 layout) or "legacy" (no _s{img} segment)
    mtime: float
    y_path: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return dict(path=self.path, rows=self.rows, dim=self.dim,
                    split_key=self.split_key, fmt=self.fmt,
                    mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.mtime)),
                    y_path=self.y_path, basename=os.path.basename(self.path))


def npy_shape(path: str) -> Tuple[int, int]:
    """Row/column count without materialising the array (memory-map the header)."""
    a = np.load(path, mmap_mode="r")
    shape = tuple(int(v) for v in a.shape)
    del a
    rows = shape[0] if shape else 0
    dim = shape[1] if len(shape) > 1 else 1
    return rows, dim


# ---------------------------------------------------------------------------
# The TTA identity of a cache file
# ---------------------------------------------------------------------------
# A cache filename records the backbone, the image size and the split folder. It
# does not record the TTA setting, so the two test caches of every four-class bed
# have identical shapes and cannot be told apart by name; row count cannot
# separate them either. Arithmetic can. Averaging k views of a vector of norm N
# whose views have mean pairwise cosine c gives
#
#     ||mean|| = N * sqrt((1 + (k - 1) c) / k)  <  N      for c < 1
#
# so within a matched pair the averaged (tta > 1) file always has the smaller mean
# row norm. Measured separation on these caches is 1.00%-2.52%, implying
# c = 0.93-0.97; two extractions of the *same* setting differ by ~1e-4 relative
# (fp16 non-determinism). The comparison is only ever used within a pair of
# same-shape files, never as an absolute test of "is this one augmented".
#
# This rule used to live only in ablation/common.py. code_final_fixed_v5.py bound
# legacy caches by row count alone and took the newest match, so a tta=1 request
# could bind the tta=3 file, a tta=3 validation request was served from the
# never-augmented Train cache -- and either way the bind log recorded the
# *requested* value. A run artifact that asserts a condition it did not honour is
# the same defect the desk rejection objected to, one level down, so the rule now
# lives here where both callers reach it.

STRICT_TTA_DEFAULT = os.environ.get("ABLATION_STRICT_TTA", "").strip().lower() \
    in ("1", "true", "yes")

# Two same-shape files are the same matrix when their mean row norms agree to
# within this relative tolerance, and a TTA pair when they do not. 0.2% sits two
# orders of magnitude above the fp16 noise floor and five times below the smallest
# real separation measured here.
NORM_TOL = 0.002

DUP_PROBE_ROWS = 256       # rows compared when confirming a same-norm pair
DUP_REL_TOL = 5e-3

TTA_SETTINGS = (1, 3)      # the only two settings any cache in this project holds


class TtaCacheUnavailable(RuntimeError):
    """The requested TTA setting has no cache and strict mode forbids substituting."""


class CacheAmbiguity(RuntimeError):
    """More distinct same-shape caches than TTA settings; refusing to guess."""


# A split with only one distinct matrix has no pair to compare against, so norm
# cannot classify it. Two rules cover every case that occurs in this project:
#
#   * train is tta=1 by construction -- code_final_fixed_v4.py:446 forces
#     `tta_n = cfg.tta if split in ["valid","test"] else 1`, so no run of v4 ever
#     wrote an augmented train cache;
#   * the legacy figshare valid cache is tta=3, established by numerical identity:
#     the legacy figshare *test* cache equals the tta=3 file of cache_pw_s224 to
#     within 1e-5..4e-4 relative (against 1.2e-1..1.7e-1 to the tta=1 file, with
#     the two train caches as a same-setting noise floor), so the whole legacy set
#     is one v4 run with cfg.tta=3 -- and such a run writes valid at cfg.tta.
#
# Anything else single-valued is reported as undetermined, never assumed. Keys are
# (source_bed, cache split word), because that is what identifies the file, not the
# bed variant asking for it: pw_identifier and pw_family read the same caches.

SINGLETON_TTA: Dict[Tuple[str, str], int] = {
    ("pw", "valid"): 3,
}


def _bed_family(bed: str) -> str:
    """'pw_family' -> 'pw', 'nick_dedup' -> 'nick'. The key rules are registered under.

    Reportability, target tables and reason strings are properties of the *cohort*,
    and a variant that only re-manifests the same folder inherits them. Cache
    location is a different question with a different answer -- see _cache_family.
    """
    b = str(bed).strip().lower()
    if b.startswith("pw"):
        return "pw"
    if b.startswith("nick"):
        return "nick"
    return b


# Where a bed's features live, which is not the same question as which rules apply.
# All four four-class beds enumerate `balance 4class`, so all four are served by
# cache_bal_s224: `nick` adopts the archive's own Train/Test folders, `nick_dedup`
# drops rows from the train side only, and `bal`/`imb` re-cut the split -- none of
# them touches a pixel the balanced caches do not already hold, and every manifest
# row carries the folder and cache index it came from. Keeping this separate from
# _bed_family is what lets a new manifest variant cost zero GPU hours.
_CACHE_FAMILY = {"bal": "bal", "imb": "bal", "nick": "bal", "nick_dedup": "bal"}


def _cache_family(bed: str) -> str:
    fam = _bed_family(bed)
    return _CACHE_FAMILY.get(str(bed).strip().lower(), _CACHE_FAMILY.get(fam, fam))


class CacheIndex:
    """Resolve (bed, backbone, split, expected_rows) to a cache file on disk.

    The lookup is deliberately dumb and explicit. It globs
    ``X_{backbone}_s{img}_{split}_*.npy`` (falling back to the pre-v4 layout
    ``X_{backbone}_{split}_*.npy``, which is what the figshare folder still holds),
    keeps only candidates whose row count equals what the manifest expects, and
    fails loudly on zero matches.

    Row count cannot separate the two test caches of a bed: they are one TTA pair
    with identical shapes. ``resolve_tta`` does, by mean row norm, and is what every
    caller that cares about the experimental condition should use. Plain ``resolve``
    is kept for callers that only ask "does a cache of the right size exist" -- the
    availability probe and ``--check-caches`` -- and records the choice as ambiguous
    when more than one file matches, rather than pretending a timestamp is a setting.
    """

    def __init__(self, cache_dirs: Sequence[str], img_size: int = 224,
                 strict_tta: Optional[bool] = None):
        self.cache_dirs = [d for d in cache_dirs if d and os.path.isdir(d)]
        self.img_size = int(img_size)
        self.strict_tta = STRICT_TTA_DEFAULT if strict_tta is None else bool(strict_tta)
        self._shape: Dict[str, Tuple[int, int]] = {}
        self._norm: Dict[str, float] = {}
        self._classified: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
        self.resolved: Dict[str, CacheRef] = {}
        self.ambiguous: Dict[str, List[str]] = {}
        # tag -> {"requested": int, "effective": int, "substituted": bool, "note": str,
        #         "file": str, "classification": str}
        self.tta_binds: Dict[str, Dict[str, Any]] = {}

    def _shape_of(self, p: str) -> Tuple[int, int]:
        if p not in self._shape:
            self._shape[p] = npy_shape(p)
        return self._shape[p]

    def mean_row_norm(self, path: str, n_rows: int = 400) -> float:
        """Mean L2 norm over the first `n_rows` rows. Cached per path.

        This is the only place a feature value is read during resolution, and it
        reads a few hundred rows, never the whole file.
        """
        if path not in self._norm:
            X = np.load(path, mmap_mode="r")
            A = np.asarray(X[: min(int(n_rows), int(X.shape[0]))], dtype=np.float64)
            del X
            self._norm[path] = float(np.linalg.norm(A, axis=1).mean())
        return self._norm[path]

    def _same_matrix(self, a: str, b: str) -> bool:
        """Cheap check that two same-shape, same-norm files hold the same features."""
        A = np.load(a, mmap_mode="r")
        Bx = np.load(b, mmap_mode="r")
        if A.shape != Bx.shape:
            return False
        k = min(DUP_PROBE_ROWS, int(A.shape[0]))
        u = np.asarray(A[:k], dtype=np.float64)
        v = np.asarray(Bx[:k], dtype=np.float64)
        del A, Bx
        d = np.linalg.norm(u - v, axis=1) / np.maximum(np.linalg.norm(u, axis=1), 1e-12)
        return bool(d.mean() < DUP_REL_TOL)

    def classify(self, backbone: str, split: str, expect_rows: int,
                 bed: str = "") -> Dict[str, Any]:
        """Which TTA settings of one source folder exist, and in which file.

        Returns ``{"by_tta": {1: CacheRef, 3: CacheRef}, "undetermined": [CacheRef],
        "duplicates": {basename: [basename]}, "rows": int, "note": str}``. Raises
        FileNotFoundError when nothing of the right size exists and CacheAmbiguity
        when more distinct same-size matrices exist than there are TTA settings.
        """
        fam = _bed_family(bed)
        ck = (backbone, split, int(expect_rows), fam)
        if ck in self._classified:
            return self._classified[ck]

        cands = [c for c in self.candidates(backbone, split) if c.rows == expect_rows]
        if not cands:
            any_rows = sorted({c.rows for c in self.candidates(backbone, split)})
            raise FileNotFoundError(
                f"no cached features for {backbone} split={split!r} with {expect_rows} "
                f"rows in {self.cache_dirs}; row counts present: {any_rows or 'none'}.")

        # Collapse re-extractions of one setting: same shape and same mean row norm is
        # the same matrix up to fp16 non-determinism, confirmed on a row sample so a
        # genuine near-tie is never swallowed.
        groups: List[List[CacheRef]] = []
        for c in sorted(cands, key=lambda r: -self.mean_row_norm(r.path)):
            for g in groups:
                ref = g[0]
                nr = self.mean_row_norm(ref.path)
                if c.dim == ref.dim and abs(self.mean_row_norm(c.path) - nr) \
                        <= NORM_TOL * max(nr, 1e-12) and self._same_matrix(c.path, ref.path):
                    g.append(c)
                    break
            else:
                groups.append([c])

        # A file with its labels beside it is the better representative: the label
        # vector is what makes the bind checkable at all.
        reps = [sorted(g, key=lambda r: (r.y_path is None, -r.mtime))[0] for g in groups]
        dups = {os.path.basename(g[0].path): [os.path.basename(x.path) for x in g[1:]]
                for g in groups if len(g) > 1}

        by_tta: Dict[int, CacheRef] = {}
        undetermined: List[CacheRef] = []
        if len(reps) == 1:
            r = reps[0]
            if split == "train":
                by_tta[1] = r
                note = ("single cache; train is tta=1 by construction "
                        "(code_final_fixed_v4.py:446 never augments the train split)")
            elif (fam, split) in SINGLETON_TTA:
                t = SINGLETON_TTA[(fam, split)]
                by_tta[t] = r
                note = (f"single cache; tta={t} established by numerical identity with the "
                        f"classified tta={t} test cache of the same extraction run")
            else:
                undetermined.append(r)
                note = "single cache and no rule fixes its TTA; treated as undetermined"
        elif len(reps) == 2:
            hi, lo = reps[0], reps[1]           # sorted by descending norm above
            nh, nl = self.mean_row_norm(hi.path), self.mean_row_norm(lo.path)
            by_tta[1], by_tta[3] = hi, lo
            note = (f"pair separated by mean row norm {nh:.3f} vs {nl:.3f} (gap "
                    f"{100.0 * (nh - nl) / max(nh, 1e-12):.2f}%); the averaged view has "
                    f"the smaller norm")
        else:
            raise CacheAmbiguity(
                f"{fam or '?'}/{backbone}/{split}: {len(reps)} distinct {expect_rows}-row "
                f"caches ({[os.path.basename(r.path) for r in reps]}), but only "
                f"{len(TTA_SETTINGS)} TTA settings are defined. Row norm orders a pair, "
                f"not a triple; move or delete the extra extractions, or re-extract this "
                f"split, rather than letting a timestamp choose.")

        out = {"by_tta": by_tta, "undetermined": undetermined, "duplicates": dups,
               "rows": int(expect_rows), "note": note}
        self._classified[ck] = out
        return out

    def resolve_tta(self, backbone: str, split: str, expect_rows: int, tta: int,
                    role: str = "", bed: str = "",
                    tag: str = "") -> Tuple[CacheRef, int, str]:
        """(ref, effective_tta, note) for one (backbone, source folder, tta) request.

        `role` is what the *request* is for, not where the rows live: a validation
        cohort carved out of Train still asks for the validation setting, and the fact
        that only un-augmented features exist for those rows is a substitution to be
        recorded -- not a silent redefinition of the request. Train is the exception,
        and it is forced to tta=1 here rather than trusted to the caller.
        """
        key = tag or f"{bed or '?'}/{backbone}/{split}/{expect_rows}"
        info = self.classify(backbone, split, expect_rows, bed=bed)
        by = info["by_tta"]
        want = 1 if str(role).strip().lower() == "train" else int(tta)

        def _record(ref: CacheRef, eff: int, note: str) -> Tuple[CacheRef, int, str]:
            self.resolved[key] = ref
            self.tta_binds[key] = dict(requested=int(want), effective=int(eff),
                                       substituted=bool(note), note=note,
                                       file=os.path.basename(ref.path),
                                       classification=info["note"])
            return ref, int(eff), note

        if want in by:
            return _record(by[want], want, "")
        if by:
            got = sorted(by)[0] if len(by) == 1 else (3 if want == 1 else 1)
            why = tta_substitution_reason(bed, role, split, want, got)
            if self.strict_tta:
                raise TtaCacheUnavailable(
                    f"{bed or '?'}/{backbone}/{role or split}: tta={want} requested but "
                    f"only tta={sorted(by)} exists for the {split!r} folder. {why} "
                    f"Unset ABLATION_STRICT_TTA (or pass strict_tta=False) to run at "
                    f"tta={got} with the substitution recorded, or extract the missing "
                    f"setting first.")
            return _record(by[got], got, why)
        if info["undetermined"]:
            raise TtaCacheUnavailable(
                f"{bed or '?'}/{backbone}/{role or split}: the only {info['rows']}-row "
                f"cache for the {split!r} folder is "
                f"{os.path.basename(info['undetermined'][0].path)}, whose TTA setting "
                f"cannot be determined -- a single file has no pair to compare norms "
                f"against and no rule fixes it. Re-extract this split so the setting is "
                f"known, rather than reporting a condition that is a guess.")
        raise FileNotFoundError(f"{bed or '?'}/{backbone}/{role or split}: no usable cache")

    def candidates(self, backbone: str, split: str) -> List[CacheRef]:
        """Every cache file that could back (backbone, split), newest first.

        The split word is searched in every case spelling that could have been written,
        because the four-class corpora name their folders `Train`/`Test` while a cache
        filename carries whatever string the extractor was handed. Glob is
        case-insensitive on Windows and case-sensitive everywhere else, so searching one
        spelling would make the same project resolve on the machine the caches were
        extracted on and fail on any other.
        """
        out: List[CacheRef] = []
        seen: set = set()
        spellings: List[str] = []
        for s in (split, str(split).lower(), str(split).capitalize()):
            if s not in spellings:
                spellings.append(s)
        for sp in spellings:
            pats = [(f"X_{backbone}_s{self.img_size}_{sp}_*.npy", f"s{self.img_size}"),
                    (f"X_{backbone}_{sp}_*.npy", "legacy")]
            for d in self.cache_dirs:
                for pat, fmt in pats:
                    for p in _glob.glob(os.path.join(d, pat)):
                        if p in seen:
                            continue
                        base = os.path.basename(p)
                        if fmt == "legacy" and f"_s{self.img_size}_" in base:
                            continue             # already matched by the first pattern
                        seen.add(p)
                        rows, dim = self._shape_of(p)
                        yp = os.path.join(os.path.dirname(p), "y_" + base[2:])
                        out.append(CacheRef(path=p, rows=rows, dim=dim, split_key=split,
                                            fmt=fmt, mtime=os.path.getmtime(p),
                                            y_path=yp if os.path.exists(yp) else None))
        out.sort(key=lambda r: -r.mtime)
        return out

    def resolve(self, backbone: str, split: str, expect_rows: int,
                prefer: Optional[str] = None, tag: str = "") -> CacheRef:
        cands = [c for c in self.candidates(backbone, split) if c.rows == expect_rows]
        key = tag or f"{backbone}/{split}/{expect_rows}"
        if not cands:
            any_rows = sorted({c.rows for c in self.candidates(backbone, split)})
            raise FileNotFoundError(
                f"no cached features for {backbone} split={split!r} with {expect_rows} rows "
                f"in {self.cache_dirs}; row counts present: {any_rows or 'none'}. "
                + ("Extract this backbone once with\n"
                   "    python code_final_fixed_v5.py --bed <bed> --stage baselines "
                   "--extract-missing\n"
                   "(only the backbones with no cache are extracted; the ones that produced "
                   "the submitted numbers stay bound to their existing caches)."
                   if not any_rows else
                   f"A cache for {backbone} exists but covers {any_rows} rows, not "
                   f"{expect_rows}. That is a corpus/cache disagreement, not a missing "
                   f"backbone -- check that the source folder has not changed size before "
                   f"re-extracting.")
                + " Not falling back to a differently sized cache.")
        if prefer:
            hit = [c for c in cands if prefer in os.path.basename(c.path)]
            if len(hit) == 1:
                self.resolved[key] = hit[0]
                return hit[0]
        if len(cands) > 1:
            self.ambiguous[key] = [os.path.basename(c.path) for c in cands]
        self.resolved[key] = cands[0]
        return cands[0]

    def load_rows(self, ref: CacheRef, rows: np.ndarray) -> np.ndarray:
        """Fancy-index a cache without holding the whole array longer than needed."""
        X = np.load(ref.path, mmap_mode="r")
        if int(X.shape[0]) != ref.rows:
            raise RuntimeError(f"{ref.path}: row count changed under us "
                               f"({X.shape[0]} now vs {ref.rows} at index time)")
        if rows.size and int(rows.max()) >= ref.rows:
            raise IndexError(f"{ref.path}: manifest asks for row {int(rows.max())} of a "
                             f"{ref.rows}-row cache. The manifest and the cache describe "
                             f"different corpora; refusing to index.")
        return np.asarray(X[rows], dtype=np.float32)

    def report(self) -> Dict[str, Any]:
        return dict(cache_dirs=list(self.cache_dirs), img_size=self.img_size,
                    strict_tta=bool(self.strict_tta),
                    resolved={k: v.to_json() for k, v in sorted(self.resolved.items())},
                    ambiguous={k: v for k, v in sorted(self.ambiguous.items())},
                    tta={k: v for k, v in sorted(self.tta_binds.items())},
                    tta_substitutions=sorted(k for k, v in self.tta_binds.items()
                                             if v["substituted"]))


def tta_substitution_reason(bed: str, role: str, source_split: str,
                            want: int, got: int) -> str:
    """Why a TTA request could not be served, in the terms that matter downstream.

    Spelled out per case rather than as one generic sentence, because the
    consequence differs: on the four-class beds the substitution touches only what
    the selection was made on, while on the patient-wise bed it leaves half of the
    TTA axis unmeasured. A reader of the bind log should not have to work that out.
    """
    fam, r = _bed_family(bed), str(role).strip().lower()
    if fam in ("bal", "imb", "nick") and r == "valid":
        return (f"The validation cohort is carved out of the {source_split!r} folder and v4 "
                f"never augments the training split, so the only features that exist for "
                f"these rows are un-augmented; serving tta={got} where tta={want} was "
                f"asked for. Validation drives threshold and hyperparameter selection "
                f"only, so this changes what was selected on, not what is reported on the "
                f"sealed test cohort -- but it does mean the selection was made at "
                f"tta={got}.")
    if fam == "pw" and r == "valid":
        return (f"Only the tta=3 validation cache exists for the patient-wise bed (the "
                f"legacy figshare extraction ran with cfg.tta=3 and no tta=1 validation "
                f"cache was ever written), so tta={want} cannot be served; using "
                f"tta={got}. For the TTA axis this leaves the tta_valid=1 half of the grid "
                f"unmeasured rather than measured.")
    return (f"tta={want} has no cache for the {source_split!r} folder of bed {fam or '?'}; "
            f"using tta={got}.")


def assemble_features(m: SplitManifest, index: CacheIndex, backbone: str,
                      source_rows_total: Optional[Dict[str, int]] = None,
                      prefer: Optional[str] = None,
                      tta: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize (X, y) for a manifest out of the caches that already exist.

    A manifest row says "this image is row 1,207 of the Train cache". So for each
    distinct source folder we resolve one cache file, fancy-index the rows we want,
    and scatter them into manifest order. A 400-per-class validation split carved
    out of a 5,600-row train cache costs one gather, not 1,600 forward passes.

    source_rows_total gives the expected row count per source folder (i.e. the
    size of the *whole* folder, not of this manifest) and is what makes the bind
    safe: the cache must be the one covering that folder in full.

    `tta` is part of the identity of the matrix, not a detail: the same images at
    tta=1 and at tta=3 are different vectors. Pass it and each block binds the file
    that actually holds that setting, with any substitution recorded in
    ``index.tta_binds``. Leaving it None keeps the old row-count-only behaviour and
    is only correct for callers that do not report a TTA condition.
    """
    blocks = m.row_blocks()
    if source_rows_total is None:
        source_rows_total = {sp: int(rows.max()) + 1 for sp, rows, _ in blocks}
    X: Optional[np.ndarray] = None
    for sp, rows, pos in blocks:
        ref = _resolve_block(m, index, backbone, sp, int(source_rows_total[sp]),
                             prefer=prefer, tta=tta)
        chunk = index.load_rows(ref, rows)
        if X is None:
            X = np.zeros((m.n, chunk.shape[1]), dtype=np.float32)
        elif chunk.shape[1] != X.shape[1]:
            raise RuntimeError(f"feature dimension differs across source folders for "
                               f"{m.bed}/{m.role}: {chunk.shape[1]} vs {X.shape[1]}")
        X[pos] = chunk
    if X is None:
        raise RuntimeError(f"manifest {m.bed}/{m.role} is empty")
    return X, m.y()


def _bind_tag(m: SplitManifest, backbone: str, source_split: str) -> str:
    """The key under which one (manifest role, backbone, source folder) bind is filed.

    Exported as a function rather than inlined because `code_final_fixed_v5.py` reads
    the same keys back out of ``index.resolved`` and ``index.tta_binds`` when it writes
    its bind log. Two spellings of one key would make that lookup miss and fall back to
    recording the *requested* TTA as if it had been honoured, which is precisely the
    defect the threading was added to remove.
    """
    return f"{m.source_bed}/{backbone}/{source_split}/{m.role}"


def _resolve_block(m: SplitManifest, index: CacheIndex, backbone: str, source_split: str,
                   expect_rows: int, prefer: Optional[str] = None,
                   tta: Optional[int] = None) -> CacheRef:
    """One source folder of one manifest -> the cache file that backs it.

    Kept in one place so features and labels cannot resolve differently: gathering
    labels from the tta=1 file while gathering features from the tta=3 file would
    make the label check pass on a matrix it never validated.

    The bookkeeping tag carries the role because two roles can draw on one folder at
    different settings -- train and valid both come out of `Train`, train always at
    tta=1 -- and a tag without the role would let the second bind overwrite the first
    in ``resolved``/``tta_binds``. Those dicts go into the run artifact, so the
    overwrite would drop the record that train bound the un-augmented file on purpose
    and leave only the validation substitution behind.
    """
    tag = _bind_tag(m, backbone, source_split)
    sk = _cache_split_key(source_split)
    if tta is None:
        return index.resolve(backbone, sk, expect_rows, prefer=prefer, tag=tag)
    ref, _eff, _note = index.resolve_tta(backbone, sk, expect_rows, int(tta),
                                         role=m.role, bed=m.source_bed, tag=tag)
    return ref


def assemble_labels(m: SplitManifest, index: CacheIndex, backbone: str,
                    source_rows_total: Optional[Dict[str, int]] = None,
                    prefer: Optional[str] = None,
                    tta: Optional[int] = None) -> Optional[np.ndarray]:
    """Gather the *cached* labels for a manifest, the same way as the features.

    This is the check that makes a legacy bind trustworthy. assemble_features returns
    m.y() as the label vector, so comparing that against m.y() proves nothing -- it is
    the same array. The cache directories also hold y_*.npy written at extraction
    time, and those labels encode the class order that was in force when the features
    were computed. If the folder order changed since (a renamed class, an added
    class), the gathered cache labels and the manifest labels disagree and the caller
    must refuse to train rather than silently learn a permuted target.

    Returns None when no y file sits beside the resolved X file, in which case the
    bind rests on the row-count assertion alone and the caller should say so.
    """
    blocks = m.row_blocks()
    if source_rows_total is None:
        source_rows_total = {sp: int(rows.max()) + 1 for sp, rows, _ in blocks}
    y = np.zeros(m.n, dtype=np.int64)
    for sp, rows, pos in blocks:
        ref = _resolve_block(m, index, backbone, sp, int(source_rows_total[sp]),
                             prefer=prefer, tta=tta)
        if not ref.y_path or not os.path.exists(ref.y_path):
            return None
        Y = np.load(ref.y_path, mmap_mode="r")
        if int(Y.shape[0]) != ref.rows:
            raise RuntimeError(f"{ref.y_path}: {Y.shape[0]} labels for a {ref.rows}-row "
                               f"feature cache")
        y[pos] = np.asarray(Y[rows], dtype=np.int64)
    return y


def _cache_split_key(source_split: str) -> str:
    """'Train' -> 'train'. Cache filenames use the lowercase split word."""
    return str(source_split).strip().lower()

def source_split_sizes(root: str, class_names: Sequence[str],
                       splits: Sequence[str]) -> Dict[str, int]:
    """Row count of each on-disk folder, i.e. the size the cache must have."""
    out: Dict[str, int] = {}
    for sp in splits:
        n = 0
        for cls in class_names:
            n += len(list_images(os.path.join(root, sp, cls)))
        out[sp] = n
    return out


def bed_source_sizes(bm: BedManifest) -> Dict[str, int]:
    used = sorted({sp for m in bm.splits.values() for sp in set(m.source_splits)})
    return source_split_sizes(bm.root, bm.class_names, used)

# ---------------------------------------------------------------------------
# Duplicate / near-duplicate audit
# ---------------------------------------------------------------------------

def content_duplicate_audit(roots: Dict[str, str], class_names: Sequence[str],
                            splits: Dict[str, Sequence[str]],
                            mode: str = "dihedral") -> Dict[str, Any]:
    """Quantify redundancy instead of calling it unquantified.

    The manuscript's limitation section currently concedes that residual overlap
    is unmeasured. It does not have to be. Byte-exact hashing over the Kaggle
    four-class corpus already gives 7,013 distinct images among 7,200, i.e. 187
    redundant copies, with no Train-Test group and no cross-class group. What byte
    hashing cannot see is a copy that was flipped or rotated, which is precisely
    what an augmented corpus is likely to contain (the meningioma class ships 100
    'Tr-aug-me' and 103 'Te-aug-me' files). mode='dihedral' closes that gap by
    hashing the orientation-canonical form.

    Returns counts plus the concrete cross-split and cross-class collisions, so a
    reviewer can check the claim rather than take it.
    """
    keys: Dict[str, str] = {}
    meta: Dict[str, Tuple[str, str, str]] = {}     # path -> (bed, split, class)
    for bed, root in roots.items():
        for sp in splits[bed]:
            for cls in class_names:
                for p in list_images(os.path.join(root, sp, cls)):
                    meta[p] = (bed, sp, cls)
    paths = sorted(meta)
    fn = {"md5": file_md5, "dihedral": dihedral_key}[mode]
    for p in paths:
        keys[p] = fn(p)

    by_key: Dict[str, List[str]] = collections.defaultdict(list)
    for p, k in keys.items():
        by_key[k].append(p)

    multi = {k: v for k, v in by_key.items() if len(v) > 1}
    cross_split, cross_class = [], []
    for k, v in multi.items():
        sps = {(meta[p][0], meta[p][1]) for p in v}
        cls = {meta[p][2] for p in v}
        if len(sps) > 1:
            cross_split.append(dict(key=k, members=[os.path.relpath(p, roots[meta[p][0]])
                                                    for p in sorted(v)],
                                    splits=sorted(f"{b}/{s}" for b, s in sps)))
        if len(cls) > 1:
            cross_class.append(dict(key=k, members=[os.path.basename(p) for p in sorted(v)],
                                    classes=sorted(cls)))
    return dict(mode=mode, n_images=len(paths), n_distinct=len(by_key),
                n_redundant_copies=len(paths) - len(by_key),
                n_groups_with_copies=len(multi),
                largest_group=max((len(v) for v in by_key.values()), default=0),
                n_cross_split_groups=len(cross_split),
                n_cross_class_groups=len(cross_class),
                cross_split_groups=cross_split[:200],
                cross_class_groups=cross_class[:200])

# ---------------------------------------------------------------------------
# Orientation-aware near-duplicate audit
# ---------------------------------------------------------------------------
# `content_duplicate_audit(mode="dihedral")` above answers an exact question: do
# two files reduce to the same orientation-canonical quantized grid. That catches a
# flipped or rotated copy re-encoded without loss. It still misses the case the
# manuscript's limitation 11 names next: a copy that was re-encoded lossily,
# contrast-stretched or slightly re-cropped lands on a different grid, so no hash of
# any canonical form collides.
#
# What follows stops hashing and starts measuring. Each image becomes a small
# grayscale grid. The grid is z-normalized, which makes it invariant to any affine
# intensity remapping a*x+b -- the transformation a re-encode or a window/level
# change applies. Two images are then compared at the maximum correlation over the
# eight dihedral transforms, which makes the comparison invariant to reflection and
# to rotation by multiples of 90 degrees. Every dihedral transform of a square grid
# is a permutation of its flattened form, so all eight comparisons cost one GEMM
# each against the same normalized matrix: no image is decoded twice and memory
# does not grow eightfold.
#
# Stated plainly so the resulting number is not oversold, this still cannot see a
# crop that changes the field of view by more than a few percent, a non-affine
# intensity transform such as gamma or histogram equalization, or a rotation by an
# angle that is not a multiple of 90 degrees. The figure it produces is a LOWER
# BOUND on residual overlap. That is the useful direction: it converts "residual
# overlap is unquantified" into "at least this many pairs cross the split, and here
# is the file list".

DIHEDRAL_NAMES = ("id", "rot90", "rot180", "rot270",
                  "flip", "flip_rot90", "flip_rot180", "flip_rot270")


def _dihedral_perms(side: int) -> List[np.ndarray]:
    """Column permutations of a flattened side x side grid, one per dihedral element.

    If x is img.reshape(-1) then x[perm[t]] is the flattened form of transform t of
    img, so a whole corpus can be reoriented with fancy indexing.
    """
    base = np.arange(side * side, dtype=np.int64).reshape(side, side)
    out: List[np.ndarray] = []
    for flip in (False, True):
        b = np.fliplr(base) if flip else base
        for r in range(4):
            out.append(np.ascontiguousarray(np.rot90(b, r).reshape(-1)))
    return out


def orientation_descriptor(path: str, side: int = 32) -> np.ndarray:
    """Flattened side x side grayscale grid. Pillow only; no torch, no imagehash."""
    from PIL import Image                                    # local import on purpose

    with Image.open(path) as im:
        g = im.convert("L").resize((side, side), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32).reshape(-1)

def _load_descriptors(paths: Sequence[str], side: int = 32,
                      progress_every: int = 1000) -> np.ndarray:
    """(n, side*side) float32 grid stack, one decode per file."""
    n, m = len(paths), side * side
    D = np.empty((n, m), dtype=np.float32)
    for i, p in enumerate(paths):
        D[i] = orientation_descriptor(p, side=side)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    decoded {i + 1}/{n}", flush=True)
    return D


def _znorm(D: np.ndarray) -> np.ndarray:
    """Row-wise zero mean, unit variance. Kills any affine intensity remap.

    A constant row (a blank image) has no direction; it is left at zero so that its
    correlation with everything is 0 rather than NaN, i.e. it is never reported as a
    duplicate of another blank. Blank slices are not the leakage this is looking for.
    """
    Z = D - D.mean(axis=1, keepdims=True)
    sd = np.sqrt((Z * Z).mean(axis=1, keepdims=True))
    flat = (sd <= 1e-6).reshape(-1)
    np.divide(Z, np.where(sd <= 1e-6, 1.0, sd), out=Z)
    Z[flat] = 0.0
    return np.ascontiguousarray(Z)


def near_duplicate_pairs(Z: np.ndarray, side: int, thresh: float = 0.98,
                         block: int = 512) -> List[Tuple[int, int, float, int]]:
    """[(i, j, best_corr, dihedral_index)] with i < j and best_corr >= thresh.

    best_corr is the maximum Pearson correlation over the eight dihedral transforms.
    The dihedral group is closed under inverse, so scanning i < j while transforming
    the right-hand side covers both directions of every pair.
    """
    n, m = Z.shape
    perms = _dihedral_perms(side)
    ZT = [np.ascontiguousarray(Z[:, p].T) for p in perms]     # (m, n) each
    out: List[Tuple[int, int, float, int]] = []
    for s in range(0, n, block):
        e = min(s + block, n)
        best = np.full((e - s, n), -2.0, dtype=np.float32)
        which = np.zeros((e - s, n), dtype=np.int8)
        for t, Zt in enumerate(ZT):
            S = (Z[s:e] @ Zt) / float(m)
            upd = S > best
            best[upd] = S[upd]
            which[upd] = t
        # row a is global index s+a, column b is global index b; keep b > s+a
        keep = np.triu(np.ones_like(best, dtype=bool), k=s + 1)
        ii, jj = np.nonzero((best >= thresh) & keep)
        for a, b in zip(ii, jj):
            out.append((s + int(a), int(b), float(best[a, b]), int(which[a, b])))
    return out

def orientation_duplicate_audit(root: str, class_names: Sequence[str],
                                splits: Sequence[str] = ("Train", "Test"),
                                side: int = 32, thresh: float = 0.98,
                                manifests: Optional[Dict[str, SplitManifest]] = None,
                                max_examples: int = 300,
                                block: int = 512) -> Dict[str, Any]:
    """Put a number on residual overlap, at the split level that matters.

    Two questions get separate answers, because they have different consequences:

      * FOLDER level -- pairs crossing the Train/Test folders. This is the number
        limitation 11 leaves open, and it bounds how optimistic the reported test
        accuracy can be.
      * ROLE level -- pairs crossing train/valid/test as the manifests define them.
        This is the one that governs selection: a train-valid pair inflates the
        validation score that fixes the thresholds and picks the configuration, and
        a train-test or valid-test pair contaminates the final number. Pass
        `manifests` (e.g. `load_bed("bal").splits`) to get it; omit for folders only.

    Also returned: pairs that cross CLASS boundaries. Those are not leakage but
    label noise, and they are worth separating -- a duplicate pair carrying two
    different labels puts a ceiling on achievable accuracy that no method can beat.

    The threshold is a correlation on a 32x32 z-normalized grid. 0.98 is strict: an
    independent slice of the same patient typically lands well below it, so the count
    should be read as near-copies rather than as same-patient pairs. Lower it to see
    how fast the count grows, which is itself informative; the report records the
    value used so the number is never quoted without its threshold.
    """
    meta: List[Tuple[str, str, str]] = []          # (split_folder, class, relpath)
    paths: List[str] = []
    for sp in splits:
        for cls in class_names:
            for p in list_images(os.path.join(root, sp, cls)):
                paths.append(p)
                meta.append((sp, cls, os.path.relpath(p, root).replace("\\", "/")))
    if not paths:
        raise RuntimeError(f"no images under {root} for splits {tuple(splits)}")

    print(f"  decoding {len(paths)} images at {side}x{side} ...", flush=True)
    Z = _znorm(_load_descriptors(paths, side=side))
    print(f"  comparing {len(paths)}x{len(paths)} over 8 orientations "
          f"(threshold {thresh}) ...", flush=True)
    pairs = near_duplicate_pairs(Z, side=side, thresh=thresh, block=block)

    role_of: Dict[str, str] = {}
    if manifests:
        for role, m in manifests.items():
            for f in m.files:
                role_of[f.replace("\\", "/")] = role

    def rec(i: int, j: int, c: float, t: int) -> Dict[str, Any]:
        return dict(a=meta[i][2], b=meta[j][2], corr=round(float(c), 4),
                    transform=DIHEDRAL_NAMES[t],
                    a_split=meta[i][0], b_split=meta[j][0],
                    a_class=meta[i][1], b_class=meta[j][1],
                    a_role=role_of.get(meta[i][2]), b_role=role_of.get(meta[j][2]))

    cross_folder, cross_class, within, cross_role = [], [], [], []
    role_counts: Dict[str, int] = collections.Counter()
    for i, j, c, t in pairs:
        r = rec(i, j, c, t)
        if r["a_split"] != r["b_split"]:
            cross_folder.append(r)
        else:
            within.append(r)
        if r["a_class"] != r["b_class"]:
            cross_class.append(r)
        ra, rb = r["a_role"], r["b_role"]
        if ra and rb:
            key = "_".join(sorted((ra, rb)))
            role_counts[key] += 1
            if ra != rb:
                cross_role.append(r)

    n_ident = sum(1 for _, _, c, _ in pairs if c >= 0.9999)
    n_reor = sum(1 for _, _, _, t in pairs if t != 0)
    covered = len(role_of) if role_of else 0
    report: Dict[str, Any] = dict(
        mode="orientation_correlation", root=os.path.basename(os.path.abspath(root)),
        side=int(side), threshold=float(thresh), splits=list(splits),
        n_images=len(paths), n_pairs=len(pairs),
        n_pairs_essentially_identical=n_ident,
        n_pairs_needing_reorientation=n_reor,
        n_cross_folder_pairs=len(cross_folder),
        n_within_folder_pairs=len(within),
        n_cross_class_pairs=len(cross_class),
        manifest_coverage=covered,
        role_pair_counts=dict(sorted(role_counts.items())),
        n_cross_role_pairs=len(cross_role),
        cross_folder_pairs=cross_folder[:max_examples],
        cross_role_pairs=cross_role[:max_examples],
        cross_class_pairs=cross_class[:max_examples],
        within_folder_pairs=within[:max_examples],
        caveat=("Lower bound. Invariant to reflection, to rotation by multiples of 90 "
                "degrees and to affine intensity change; NOT invariant to cropping, "
                "gamma/histogram remapping, or off-axis rotation. Absence of a pair "
                "here is not proof of disjointness."))
    if manifests and not role_of:
        report["warning"] = "manifests supplied but no file matched; check the root"
    return report

def orientation_threshold_sweep(root: str, class_names: Sequence[str],
                                splits: Sequence[str] = ("Train", "Test"),
                                side: int = 32,
                                thresholds: Sequence[float] = (0.999, 0.99, 0.98, 0.95, 0.90),
                                manifests: Optional[Dict[str, SplitManifest]] = None,
                                block: int = 512) -> Dict[str, Any]:
    """How fast the count grows as the threshold relaxes. One decode, one scan.

    A single number at a single threshold invites the objection that the threshold
    was chosen to produce it. The sweep answers that objection directly: pairs are
    found once at the loosest threshold and re-counted at each stricter one, so the
    whole curve costs the same as one audit. A curve that is flat from 0.999 down to
    0.98 and then climbs says the strict count is real and the loose one is picking
    up genuine anatomical similarity between different slices.
    """
    meta: List[Tuple[str, str, str]] = []
    paths: List[str] = []
    for sp in splits:
        for cls in class_names:
            for p in list_images(os.path.join(root, sp, cls)):
                paths.append(p)
                meta.append((sp, cls, os.path.relpath(p, root).replace("\\", "/")))
    if not paths:
        raise RuntimeError(f"no images under {root} for splits {tuple(splits)}")
    lo = float(min(thresholds))
    print(f"  decoding {len(paths)} images at {side}x{side} ...", flush=True)
    Z = _znorm(_load_descriptors(paths, side=side))
    print(f"  scanning once at the loosest threshold {lo} ...", flush=True)
    pairs = near_duplicate_pairs(Z, side=side, thresh=lo, block=block)

    role_of: Dict[str, str] = {}
    if manifests:
        for role, m in manifests.items():
            for f in m.files:
                role_of[f.replace("\\", "/")] = role

    rows = []
    for th in sorted({float(t) for t in thresholds}, reverse=True):
        sel = [(i, j, c, t) for i, j, c, t in pairs if c >= th]
        xf = sum(1 for i, j, _, _ in sel if meta[i][0] != meta[j][0])
        xc = sum(1 for i, j, _, _ in sel if meta[i][1] != meta[j][1])
        xr = 0
        if role_of:
            for i, j, _, _ in sel:
                ra, rb = role_of.get(meta[i][2]), role_of.get(meta[j][2])
                if ra and rb and ra != rb:
                    xr += 1
        rows.append(dict(threshold=th, n_pairs=len(sel), n_cross_folder=xf,
                         n_cross_role=(xr if role_of else None), n_cross_class=xc,
                         n_reoriented=sum(1 for _, _, _, t in sel if t != 0)))
    return dict(mode="orientation_threshold_sweep", side=int(side),
                root=os.path.basename(os.path.abspath(root)), splits=list(splits),
                n_images=len(paths), manifest_coverage=len(role_of), rows=rows,
                caveat=("Counts at each threshold come from one scan at the loosest "
                        "value, so they are nested by construction."))


# ---------------------------------------------------------------------------
# Project layout + CLI
# ---------------------------------------------------------------------------
# Bed keys are the same ones the ablation harness uses, so cache folder names
# (cache_bal_s224 / cache_pw_s224) keep working unchanged. There is no
# cache_imb_s224: `imb` resolves to the balanced cache through _CACHE_FAMILY, and a
# directory of that name -- a retired 4,200/1,200 extraction -- no longer exists and
# is named by no code path. Note the corpus folder names are misspelled on disk
# ("imbanlce 4class"); they are quoted verbatim here rather than corrected, because
# renaming folders would invalidate paths in scripts the user still runs. That
# folder is inert in any case: `imb`'s manifest records its root as `balance
# 4class`, like every other four-class bed.
#
# `nick`, `nick_dedup`, `bal` and `imb` all point at the SAME folder, and that is
# deliberate. `balance 4class` is the extracted Kaggle archive (see the comment
# above NICK_PUBLISHED), so there is no second corpus to download and no
# `nickparvar 4class` to create. What separates the four beds is the manifest, not
# the pixels: `nick` adopts the archive's own Train/Test folders, `nick_dedup`
# drops leaked rows from Train only, and `bal`/`imb` re-cut the split the way the
# retired tables did.
#
# The practical consequence is that the existing caches already cover every one of
# them. `nick`'s train rows enumerate `balance 4class/Train` in exactly the order
# cache_bal_s224's 5,600-row train matrix was written in -- same root, same folder,
# same list_images -- so no re-extraction is needed for any four-class bed.

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_FOUR_CLASS_CORPUS = os.path.join(PROJECT_ROOT, "balance 4class")

BED_ROOTS = {
    "bal": _FOUR_CLASS_CORPUS,
    "imb": os.path.join(PROJECT_ROOT, "imbanlce 4class"),
    "pw":  os.path.join(PROJECT_ROOT, "figshare"),
    "nick": _FOUR_CLASS_CORPUS,
    "nick_dedup": _FOUR_CLASS_CORPUS,
}

# ---------------------------------------------------------------------------
# The two corpora are one corpus: measured, 2026-09-05
# ---------------------------------------------------------------------------
# BED_ROOTS names two folders, and the manuscript treated them as two datasets: a
# three-class patient-wise bed over `figshare` and a four-class folder-split bed
# over `balance 4class`, reported side by side so that agreement between them
# reads as agreement across cohorts. It does not. 3,038 of figshare's 3,064
# slices -- 99.2% -- are the same picture as a file in `balance 4class`, almost
# always mirrored. figshare is a subset of the four-class corpus, not a second
# corpus, and 5,014 of the 7,200 four-class files being exactly 512x512 (figshare's
# native size) was the hint that started this.
#
# Measured with the rule the deduplicated bed already uses (DEDUP_RULE_ID),
# imported and not re-typed. That rule was fixed a priori for a different question,
# so this verdict could not have been tuned toward it, and the two answers cannot
# disagree about what "the same picture" means.
#
# WHAT IS AND IS NOT BROKEN. Each bed is internally clean and stays so: figshare's
# split is patient-disjoint (patient_overlap = 0 across all three role pairs) and
# the four-class beds carve validation out of Train. Nothing here is a within-bed
# leak. The hazard is BETWEEN beds, and it has two shapes:
#   * a table. Quoting a pw row beside a nick row as two cohorts overstates the
#     evidence: 2,334 of the 3,038 shared pictures sit in the four-class TRAIN
#     folder, so the two "cohorts" are largely the same pictures in different roles.
#   * a decision. 588 of figshare's 781 test slices (75.3%) are in the four-class
#     Train folder, and 523 of the 1,600 four-class test files (32.7%; 43.6% of the
#     1,200 in the three classes figshare has) are in figshare's train+valid pool.
#     A choice informed by one bed's test result has therefore touched pictures the
#     other bed trains on, so rejection ground (2) is answered PER BED and never
#     once for the pair. Neither bed's own sealed cohort is compromised by this --
#     each ledger still holds -- but the pair is not two independent replications.
#
# Like CHANNEL_AUDIT and TTA_WEDGE_AUDIT this is a MEASUREMENT and not a setting:
# no tensor, rule id, cache key, manifest or split reads it (asserted by the
# checker), so correcting a transcription error here cannot invalidate a cached
# matrix or move a single file between roles.
CORPUS_OVERLAP: Dict[str, Any] = {
    "rule": DEDUP_RULE_ID,
    "direction": "each figshare slice -> some file of `balance 4class`",
    "direction_note": ("the cheap direction (figshare is the smaller side) and the one "
                       "the manuscript question needs. The dihedral group is closed "
                       "under inverse, so the verdict is symmetric even though the "
                       "transform label is not"),
    "n_figshare": 3064,
    "n_four_class": 7200,
    "n_matched": 3038,
    "n_unmatched": 26,
    "n_label_contradictions": 0,
    "per_folder_fields": ("n", "matched", "into_four_train", "into_four_test"),
    "per_folder": {
        "train/glioma":     (850, 840, 685, 155),
        "train/meningioma": (411, 410, 322,  88),
        "train/pituitary":  (550, 550, 364, 186),
        "valid/glioma":     (206, 203, 187,  16),
        "valid/meningioma": (132, 132, 101,  31),
        "valid/pituitary":  (134, 134,  87,  47),
        "test/glioma":      (370, 358, 302,  56),
        "test/meningioma":  (165, 165, 130,  35),
        "test/pituitary":   (246, 246, 156,  90),
    },
    # HOW the match was found. All-identity would mean a straight file copy; a spread
    # over the dihedral group means a re-release with an orientation convention of its
    # own. This is neither: it is one flip, near-universally.
    "transforms": {"flip": 3030, "identity": 8},
    "corr_min": 0.999618,
    "corr_median": 0.999920,
    "n_corr_exactly_one": 0,
    "n_byte_identical_grey_planes": 0,
    "provenance_reading": ("3,030 of 3,038 through a single horizontal flip, no "
                           "correlation of exactly 1.0, and not one pair whose native "
                           "grey planes are byte-identical -- the signature of a mirrored "
                           "re-encode of the whole set, not of copied files. Consistent "
                           "with the four-class archive having been assembled from "
                           "figshare + SARTAJ + Br35H, which is also why `notumor` (a "
                           "class figshare does not contain) matched 0 of 1,800"),
    # COVERAGE of the four-class side, which is what makes "subset" the right word:
    # every shared picture is one figshare slice and one four-class file, and the
    # 4,162 unmatched four-class files are the notumor class plus the SARTAJ residue.
    "n_distinct_four_matched": 3038,
    "coverage_four": {"Train/glioma": (1174, 1400), "Train/meningioma": (553, 1400),
                      "Train/notumor": (0, 1400), "Train/pituitary": (607, 1400),
                      "Test/glioma": (227, 400), "Test/meningioma": (154, 400),
                      "Test/notumor": (0, 400), "Test/pituitary": (323, 400)},
    # ---- the four controls, each answering a way the count could have been wrong ----
    # Reported as numbers rather than as "verified", because a control whose result is
    # not in the record is indistinguishable from a control that was never run.
    "controls": {
        # (a) promiscuity, negative control: the one class figshare does not contain
        "notumor_matched": (0, 1800),
        # (b) injectivity: one figshare slice cannot legitimately be two four-class files
        "injective": True, "n_distinct_targets": 3038, "n_collisions": 0,
        # (c) the flip is real, not bilateral symmetry. A brain is approximately
        # mirror-symmetric, so `flip` could have been a property of the anatomy. Scored
        # 400 pairs with and without the matched transform: median 0.9999 against
        # 0.7135, and not one pair reaches the 0.9995 floor untransformed.
        "symmetry_n": 400, "symmetry_median_with": 0.9999,
        "symmetry_median_without": 0.7135, "symmetry_n_pass_untransformed": 0,
        # (d) grid saturation: the decision is taken on a 128x128 grid, so it was
        # re-checked at NATIVE resolution for all 3,038 pairs (not a sample).
        "native_census_n": 3038, "native_same_dims": 3038,
        "native_resid_median": 0.913, "native_resid_p99": 1.268,
        "native_resid_max": 3.371, "native_n_over_budget": 9,
        "native_over_budget_note": ("9 pairs (0.30%) exceed the rule's 3.0-grey-level "
                                    "budget at native resolution while passing at "
                                    "128x128, and all nine are contiguous slices of one "
                                    "figshare patient (train/pituitary/108931__*). "
                                    "Listed rather than rounded away: a subset claim of "
                                    "99.2% and a subset claim of 98.9% lead to the same "
                                    "conclusion, so the exception list is cheaper than "
                                    "the argument"),
        # (e) the 26 unmatched slices are absences, not near-misses under the threshold
        "unmatched_best_screen": 0.9623, "unmatched_n_screening_over_0_99": 0,
    },
    # ---- the exposure the manuscript has to disclose --------------------------------
    "exposure": {
        "fig_test_in_four_train": (588, 781),
        "four_test_in_fig_dev": (523, 1600),
        "four_test_in_fig_dev_three_class": (523, 1200),
        "cross_split": {"train->Train": 1371, "train->Test": 429,
                        "valid->Train": 375, "valid->Test": 94,
                        "test->Train": 588, "test->Test": 181},
    },
    "audited_by": ("measured 2026-09-05 over every file of both corpora (no sampling), "
                   "using same_picture / DEDUP_* imported from this module; native-"
                   "resolution census over all 3,038 matched pairs"),
}

# Which side of the overlap each bed sits on. Not derived from _bed_family: that
# function answers "whose rules apply", and a bed could one day be added that shares
# rules with `nick` while reading a corpus this audit never looked at. An unknown bed
# gets no claim rather than an inherited one.
_OVERLAP_SIDE = {"pw": "figshare", "nick": "four_class", "bal": "four_class",
                 "imb": "four_class"}


def corpus_overlap_side(bed: str) -> str:
    """'figshare', 'four_class', or '' when this bed's corpus was never audited."""
    return _OVERLAP_SIDE.get(_bed_family(bed), "")


def cohort_overlap(bed: str) -> Dict[str, Any]:
    """Columns saying how much of THIS bed's test cohort the other bed trains on.

    Stamped into every artifact and CSV row beside `reportability`, and for the same
    reason: two years from now the surviving object is a file, and a reader comparing
    a pw row against a nick row has to be able to see from the rows themselves that
    the two cohorts are 99.2% the same pictures. A sentence in a comment here would
    not survive the copy into a table.

    `test_in_other_dev_frac` is the number that bears on rejection ground (2), and it
    is deliberately the bed's OWN exposure rather than the symmetric overlap: what a
    reader needs is not "these corpora intersect" but "this much of the cohort I am
    being shown was in the other experiment's training pool". Empty strings, not
    zeros, for a bed whose corpus was not audited -- 0.0 would read as "measured, and
    clean", which is a different claim from "not measured".
    """
    side = corpus_overlap_side(bed)
    if not side:
        return dict(corpus_shared_with="", cohort_test_in_other_dev="",
                    cohort_test_in_other_dev_frac="", cohort_overlap_note="")
    ex = CORPUS_OVERLAP["exposure"]
    if side == "figshare":
        n, tot = ex["fig_test_in_four_train"]
        other = "balance 4class (beds nick / nick_dedup / bal / imb)"
    else:
        n, tot = ex["four_test_in_fig_dev"]
        other = "figshare (beds pw_identifier / pw_family)"
    return dict(corpus_shared_with=other,
                cohort_test_in_other_dev=f"{n}/{tot}",
                cohort_test_in_other_dev_frac=round(n / tot, 4),
                cohort_overlap_note=(
                    f"{100 * CORPUS_OVERLAP['n_matched'] / CORPUS_OVERLAP['n_figshare']:.1f}%"
                    f" of figshare is the same picture as a file in `balance 4class`, so "
                    f"this bed and the beds over {other.split(' (')[0]} are not "
                    f"independent cohorts; {n} of this bed's {tot} test files are in that "
                    f"corpus's development pool. This bed's own split is unaffected."))


def corpus_overlap_note() -> str:
    """One paragraph for --stage dry-run, the artifact, and the manuscript's cohorts.

    Built from CORPUS_OVERLAP so it cannot drift from the numbers, and it leads with
    the count rather than the controls: the controls matter only because the count is
    large enough to change what the paper may claim.
    """
    o = CORPUS_OVERLAP
    c, ex = o["controls"], o["exposure"]
    ft, ftn = ex["fig_test_in_four_train"]
    tt, ttn = ex["four_test_in_fig_dev"]
    return (
        f"measured corpus overlap ({o['audited_by']}): {o['n_matched']} of "
        f"{o['n_figshare']} figshare slices "
        f"({100 * o['n_matched'] / o['n_figshare']:.1f}%) are the same picture as a file "
        f"in `balance 4class` under the a-priori dedup rule {o['rule']}, with "
        f"{o['n_label_contradictions']} label contradictions. figshare is therefore a "
        f"SUBSET of the four-class corpus and not a second cohort. The match is "
        f"{o['transforms'].get('flip', 0)}x a single horizontal flip against "
        f"{o['transforms'].get('identity', 0)}x identity, median correlation "
        f"{o['corr_median']:.6f}, {o['n_corr_exactly_one']} correlations of exactly 1.0 "
        f"and {o['n_byte_identical_grey_planes']} byte-identical grey planes -- a "
        f"mirrored re-encode, not copied files. Controls: notumor, the class figshare "
        f"lacks, matched {c['notumor_matched'][0]} of {c['notumor_matched'][1]}; the map "
        f"is injective ({c['n_distinct_targets']} slices to "
        f"{c['n_distinct_targets']} distinct files, {c['n_collisions']} collisions); the "
        f"flip is real rather than bilateral symmetry ({c['symmetry_n_pass_untransformed']}"
        f" of {c['symmetry_n']} pairs reach the correlation floor untransformed, median "
        f"{c['symmetry_median_without']:.4f} against "
        f"{c['symmetry_median_with']:.4f} with the flip); at native resolution all "
        f"{c['native_census_n']} pairs share dimensions with median residual "
        f"{c['native_resid_median']:.2f} grey levels and only "
        f"{c['native_n_over_budget']} over the {DEDUP_MAX_RESIDUAL:g}-level budget; and "
        f"the {o['n_unmatched']} unmatched slices are absences, not near-misses (best "
        f"screen {c['unmatched_best_screen']:.4f}, "
        f"{c['unmatched_n_screening_over_0_99']} above 0.99). The consequence for the "
        f"manuscript is a disclosure, not a re-split: each bed's own cohort is intact "
        f"(figshare is patient-disjoint, the four-class beds carve validation from "
        f"Train), but {ft} of {ftn} figshare test slices sit in the four-class Train "
        f"folder and {tt} of {ttn} four-class test files sit in figshare's train+valid "
        f"pool, so independence of the final test cohort is a per-bed claim and a pw row "
        f"beside a nick row is not two independent replications.")


def cross_bed_independence(beds: Sequence[str]) -> Dict[str, Any]:
    """Whether a table quoting exactly THESE beds may call them independent cohorts.

    The overlap does not demote any bed -- `pw` and `nick` are each reportable, each
    has an intact split, and neither number is wrong. What is wrong is a sentence, and
    only when both sides appear together: "the method was validated on two independent
    datasets". So the check has to be over the SET of beds in one table rather than
    per bed, which is why it is not part of `reportability` and cannot be.

    Returns spans_both_corpora plus the sentence to put under the table. Callers that
    build a multi-bed table should stamp this into the artifact; a caller quoting one
    bed gets spans_both_corpora=False and an empty sentence, because for a single bed
    there is nothing to overstate.
    """
    sides = {corpus_overlap_side(b) for b in beds}
    sides.discard("")
    both = {"figshare", "four_class"} <= sides
    o = CORPUS_OVERLAP
    ex = o["exposure"]
    return dict(
        beds=list(beds),
        spans_both_corpora=both,
        independent_cohorts=not both,
        independence_caveat=("" if not both else (
            f"These beds do not constitute independent cohorts. {o['n_matched']} of "
            f"{o['n_figshare']} figshare slices ({100 * o['n_matched'] / o['n_figshare']:.1f}%) "
            f"are the same picture as a file in `balance 4class`, mostly mirrored, so the "
            f"rows below are largely the same images in different roles: "
            f"{ex['fig_test_in_four_train'][0]} of {ex['fig_test_in_four_train'][1]} "
            f"figshare test slices are in the four-class Train folder and "
            f"{ex['four_test_in_fig_dev'][0]} of {ex['four_test_in_fig_dev'][1]} "
            f"four-class test files are in figshare's train+valid pool. Each bed's own "
            f"split remains valid and each is reported on its own terms; agreement "
            f"between them is not cross-dataset replication.")))

# Which beds a paper table may quote, and why the others may not be quoted.
#
# This is not a style preference. `bal` and `imb` cannot answer rejection ground
# (1) even in principle -- not because their corpus is unpublished (it is the
# published archive; see the comment above NICK_PUBLISHED) but because their
# *split* is ours. `bal` re-cuts 1,000/400/400 per class out of folders the
# archive ships as 1,400/400, so the cohort a reader would have to reproduce is a
# seed and a rule in this file rather than a folder anyone else has. `imb` is
# worse: the on-disk `imbanlce 4class` folder is never read (both manifests record
# `root = balance 4class`), its per-class training targets are hardcoded here with
# no published basis, and the folder itself turns out to be a hand-made contiguous
# tail-slice of the balanced one. The code paths stay runnable -- they are the
# diagnostic that explains the retired figures -- but a reported number needs a
# bed whose cohort someone else has also used.
REPORTABLE_BEDS = ("pw_identifier", "pw_family", "nick", "nick_dedup")

NON_REPORTABLE_BEDS: Dict[str, str] = {
    "bal": ("the corpus is the published archive but the split is not: validation and "
            "training are re-cut at 1,000/400 per class out of folders the archive "
            "ships as 1,400 train + 400 test, so the cohort a reader would have to "
            "reproduce is a seed and a rule in this file rather than a folder anyone "
            "else has. Use 'nick' for the published split, or 'nick_dedup' for the "
            "leakage-controlled variant of it"),
    "imb": ("the imbalance ratio is hardcoded with no published basis, the on-disk "
            "'imbanlce 4class' folder is never read, and that folder is itself a "
            "hand-made tail-slice of the balanced corpus"),
}


def bed_is_reportable(bed: str) -> bool:
    return _bed_family(bed) not in NON_REPORTABLE_BEDS


def non_reportable_reason(bed: str) -> str:
    """Why this bed may not carry a reported number. Empty string when it may."""
    return NON_REPORTABLE_BEDS.get(_bed_family(bed), "")


# -- the one escape hatch, and the only way to open it ------------------------
# bal and imb have to stay *runnable*: they are how the retired tables remain
# explainable, and a figure you can still recompute can be retired on the record
# while a figure whose recipe has been deleted can only be quietly dropped. But
# "runnable" and "runs by accident" are different things. Two years from now the
# only artifact left will be a CSV in out/ or runs_v5/, and the danger is not that
# someone re-derives a retired number on purpose -- it is that someone reads one
# back out of a file without remembering that the cohort was demoted.
#
# So a diagnostic run has to say so, out loud, before it starts. The declaration
# is deliberately awkward: an environment variable or an explicit call, never a
# default, never inferred from the bed name. It is recorded in the ledger notes
# and stamped into every row the ablation harness writes, so the number and its
# licence travel together in the file rather than in someone's memory.
DIAGNOSTIC_ENV = "ARTICLE2_ALLOW_NON_REPORTABLE"

_DIAGNOSTIC_DECLARED: List[str] = []


def declare_diagnostic_run(why: str) -> None:
    """Announce, in-process, that this run re-derives a retired figure.

    For scripts whose entire purpose is provenance -- verify_reproduction.py,
    verify_beta.py -- which would otherwise have to be launched with an
    environment variable set by hand every time. Prints the reason, because a
    declaration nobody sees is a default.
    """
    if why not in _DIAGNOSTIC_DECLARED:
        _DIAGNOSTIC_DECLARED.append(why)
        print(f"[diagnostic run] non-reportable beds allowed: {why}\n"
              f"                 nothing measured in this run may be quoted in a table.",
              flush=True)


def diagnostic_run_declared() -> str:
    """Why non-reportable beds are permitted here, or '' if they are not."""
    if _DIAGNOSTIC_DECLARED:
        return "; ".join(_DIAGNOSTIC_DECLARED)
    env = os.environ.get(DIAGNOSTIC_ENV, "").strip()
    if env and env.lower() not in ("0", "false", "no"):
        return f"{DIAGNOSTIC_ENV}={env}"
    return ""


def assert_bed_reportable(bed: str, what: str) -> str:
    """Refuse `what` on a non-reportable bed unless the run declared itself.

    Returns the declaration string when the bed is non-reportable and the run is
    a declared diagnostic (the caller should record it), and '' when the bed is
    reportable and nothing needs saying.
    """
    why = non_reportable_reason(bed)
    if not why:
        return ""
    decl = diagnostic_run_declared()
    if decl:
        return decl
    raise NonReportableBed(
        f"{what} on bed {bed!r}, which is not reportable: {why}.\n"
        f"Reportable beds: {list(REPORTABLE_BEDS)}.\n"
        f"If you mean to re-derive a retired figure, say so and it will run:\n"
        f"    set {DIAGNOSTIC_ENV}=1        (cmd)   -- or RUN_ALL.bat /diag\n"
        f"Whatever comes out is diagnostic and may not be quoted in a table.")

SPLIT_DIR = os.path.join(PROJECT_ROOT, "splits_v5")

CACHE_ROOT = os.path.join(PROJECT_ROOT, "ablation", "caches")


def cache_dirs_for(bed: str) -> List[str]:
    """Where features for a bed may live. Order is search order, not priority."""
    src = _cache_family(bed)         # imb / nick / nick_dedup all reuse the bal caches
    out = [os.path.join(CACHE_ROOT, f"cache_{src}_s224")]
    if src == "pw":
        out.append(BED_ROOTS["pw"])                    # legacy X_{bb}_{split}_*.npy live here
    return out


def manifest_path(bed: str, group_unit: Optional[str] = None) -> str:
    suffix = f"_{group_unit}" if group_unit else ""
    return os.path.join(SPLIT_DIR, f"split_manifest_{bed}{suffix}.json")


def build_all(bal_root: Optional[str] = None, pw_root: Optional[str] = None,
              nick_root: Optional[str] = None,
              out_dir: Optional[str] = None, seed: int = 2026,
              content_group: str = "md5", pw_seed: int = 42,
              nick_valid_frac: float = 0.15,
              require_nick: bool = False,
              build_dedup: bool = True,
              write: bool = True) -> Dict[str, BedManifest]:
    """Build every bed manifest this project needs and (optionally) write them.

    Six manifests come out:
      nick                      -- the published four-class corpus on its published
                                   Train/Test split, validation carved out of Train.
                                   Comparable to prior work by construction, and it
                                   inherits prior work's train/test overlap.
      nick_dedup                -- the same corpus and the SAME test cohort, with
                                   every training row that duplicates a test row
                                   removed. Reported beside nick, never instead of
                                   it: the pair is the leakage measurement.
      pw_identifier             -- the figshare split exactly as it sits on disk
      pw_family                 -- the same slices regrouped so that identifiers
                                   sharing a numeric stem cannot span roles
      bal, imb                  -- retained and runnable, marked non-reportable
                                   (see NON_REPORTABLE_BEDS for why)

    The two pw beds and the four four-class beds are NOT two independent cohorts:
    99.2% of figshare is the same picture, usually mirrored, as a file in
    `balance 4class` (see CORPUS_OVERLAP, measured over every file of both). Each
    bed's own split is intact and each may be reported; what may not be said is
    that agreement between a pw row and a nick row is cross-dataset replication.
    `cross_bed_independence()` produces the caveat for a table that quotes both.

    All four four-class beds read `balance 4class`, which is the extracted archive;
    there is no second corpus to fetch. `nick_dedup` is skipped with a printed
    reason if its mask cannot be built, so the other five still write.

    Set build_dedup=False to skip the deduplicated bed. Its first build decodes
    ~7,200 images on the CPU; afterwards the mask is cached and reused.

    content_group applies to the validation carve. 'strict' -- connected components
    of DEDUP_RULE_ID -- reaches the two nick beds only; bal and imb are held at 'md5'
    with a printed note, because they are non-reportable and their feature caches are
    already bound to the carve they have.
    """
    bal_root = bal_root or BED_ROOTS["bal"]
    pw_root = pw_root or BED_ROOTS["pw"]
    nick_root = nick_root or BED_ROOTS["nick"]
    out_dir = out_dir or SPLIT_DIR
    beds: Dict[str, BedManifest] = {}

    # bal and imb keep their original grouping whatever the caller asks for, and this
    # is a compatibility decision rather than an oversight. Both are non-reportable,
    # so tightening their carve buys nothing quotable; against that, every feature
    # cache on disk was extracted against their current manifests, and re-grouping
    # would move rows between train and valid and invalidate all of them. 'strict'
    # would also raise inside build_four_class_beds, which takes no relation -- so
    # asking for it here would abort the build of the four beds that DO benefit.
    legacy_cg = "md5" if content_group == "strict" else content_group
    if legacy_cg != content_group:
        print(f"  note: beds bal/imb keep content_group='{legacy_cg}'; "
              f"'{content_group}' applies to nick and nick_dedup, the beds whose "
              f"validation carve it can actually protect (bal/imb are non-reportable "
              f"and their caches are already bound to the existing carve).", flush=True)

    if os.path.isdir(nick_root):
        beds["nick"] = build_published_four_class_bed(
            nick_root, bed="nick", seed=seed, valid_frac=nick_valid_frac,
            content_group=content_group, out_dir=out_dir)
        if build_dedup:
            try:
                beds["nick_dedup"] = build_deduplicated_four_class_bed(
                    nick_root, bed="nick_dedup", source_bed_manifest=beds["nick"],
                    seed=seed, valid_frac=nick_valid_frac,
                    content_group=content_group, out_dir=out_dir,
                    mask_path=os.path.join(out_dir, DEDUP_MASK_NAME.format(bed="nick")))
                assert_test_cohorts_agree(beds["nick"], beds["nick_dedup"])
            except (RuntimeError, OSError, ValueError) as e:
                print(f"[skip] nick_dedup: {e}\n"
                      f"        The published split still builds; the leakage-controlled "
                      f"variant does not, so only the standard number is available and "
                      f"the overlap stays an unquantified limitation.\n", flush=True)
    elif require_nick:
        raise FileNotFoundError(
            f"no corpus at {nick_root}. This should be the extracted four-class archive, "
            f"so that {os.path.join(nick_root, 'Train')} and "
            f"{os.path.join(nick_root, 'Test')} exist.")
    else:
        print(f"[skip] nick: no corpus at {nick_root}; the only four-class beds a table "
              f"may quote are therefore unavailable.\n")

    four = build_four_class_beds(bal_root, seed=seed, content_group=legacy_cg)
    beds["bal"] = four["bal"]
    beds["imb"] = four["imb"]

    beds["pw_identifier"] = build_patient_bed(pw_root, group_unit="identifier",
                                              seed=pw_seed, reuse_existing_folders=True,
                                              strict_targets=True)
    beds["pw_family"] = build_patient_bed(pw_root, group_unit="family",
                                          seed=pw_seed, reuse_existing_folders=True,
                                          strict_targets=False)

    for key, bm in beds.items():
        why = non_reportable_reason(key)
        bm.audit["reportable"] = not why
        if why:
            bm.audit["not_reportable_because"] = why

    if write:
        os.makedirs(out_dir, exist_ok=True)
        for key, bm in beds.items():
            bm.save(os.path.join(out_dir, f"split_manifest_{key}.json"))
    return beds


ALL_BEDS = ("nick", "nick_dedup", "bal", "imb", "pw_identifier", "pw_family")


def load_bed(bed: str, out_dir: Optional[str] = None) -> BedManifest:
    """Load a written manifest. `bed` is one of ALL_BEDS.

    'pw' is accepted as an alias for 'pw_identifier' so existing call sites keep
    working; the family variant has to be asked for by name, because silently
    swapping the grouping unit would change what the reported number means.
    """
    if bed == "pw":
        bed = "pw_identifier"
    p = os.path.join(out_dir or SPLIT_DIR, f"split_manifest_{bed}.json")
    if not os.path.exists(p):
        hint = ""
        if bed.startswith("nick") and not os.path.isdir(BED_ROOTS["nick"]):
            hint = (f"\nThe corpus itself is missing too: the four-class beds all read the "
                    f"extracted archive at {BED_ROOTS['nick']}, which must hold Train/ + "
                    f"Test/ subfolders.")
        elif bed == "nick_dedup":
            # Distinguish "never built" from "built and refused": build_all prints a
            # reason and continues when the mask cannot be made, so the absence of
            # this one manifest is not by itself a build failure.
            hint = (f"\n'{bed}' is the leakage-controlled variant of 'nick'. It is built by "
                    f"the same --build, but it is skipped with a printed reason if its "
                    f"duplicate mask cannot be produced; check that run's output, or "
                    f"rebuild the mask alone with:  python splits_v5.py --dedup-mask")
        raise FileNotFoundError(
            f"no manifest at {p}. Build the splits first:  python splits_v5.py --build{hint}")
    return BedManifest.load(p)


def _cli() -> int:
    ap = argparse.ArgumentParser(description="build and inspect the v5 split manifests")
    ap.add_argument("--build", action="store_true", help="write the manifests")
    ap.add_argument("--show", action="store_true", help="print manifests already written")
    ap.add_argument("--check-caches", action="store_true",
                    help="resolve each manifest against the .npy caches on disk (reads headers only)")
    ap.add_argument("--dup-audit", choices=["md5", "dihedral"], default=None,
                    help="run the exact duplicate audit over the Kaggle four-class corpus")
    ap.add_argument("--near-dup-audit", action="store_true",
                    help="orientation-aware near-duplicate audit: correlation over the 8 "
                         "dihedral transforms of a z-normalized grid, reported at folder "
                         "level and (if manifests exist) at train/valid/test role level")
    ap.add_argument("--near-dup-thresh", type=float, default=0.98,
                    help="correlation threshold for --near-dup-audit (default 0.98)")
    ap.add_argument("--near-dup-side", type=int, default=32,
                    help="grid side for --near-dup-audit (default 32)")
    ap.add_argument("--near-dup-sweep", action="store_true",
                    help="with --near-dup-audit, also report the count at 0.999/0.99/0.98/"
                         "0.95/0.90 so the number is not hostage to one threshold")
    ap.add_argument("--near-dup-bed", choices=["bal", "imb", "pw", "nick", "nick_dedup"],
                    default="bal",
                    help="corpus for --near-dup-audit (default bal, the four-class corpus "
                         "limitation 11 is about)")
    ap.add_argument("--near-dup-gate", type=float, default=None,
                    help="exit 3 when the audit finds more than this many cross-ROLE "
                         "near-duplicate pairs (train/valid/test as the manifests define "
                         "them). Use 0 to demand a clean bed. Without it the audit is a "
                         "side artifact nobody has to look at.")
    ap.add_argument("--dedup-mask", action="store_true",
                    help="build (or reuse) the train-side duplicate mask for the "
                         "deduplicated four-class bed and print what it removes, without "
                         "writing any manifest. Reads image files only -- no cache, no "
                         "labels, no test read.")
    ap.add_argument("--leakage-audit", action="store_true",
                    help="count cross-ROLE duplicate pairs under the removal rule itself "
                         "(DEDUP_RULE_ID), not under the 0.98 correlation audit. This is "
                         "the criterion a bed can actually be required to pass: the "
                         "deduplicated bed satisfies it at zero by construction. Writes "
                         "splits_v5/leakage_audit_<bed>.json.")
    ap.add_argument("--leakage-bed", choices=["bal", "imb", "nick", "nick_dedup"],
                    default="nick_dedup",
                    help="bed whose manifests define the roles for --leakage-audit "
                         "(default nick_dedup, the bed the gate is meant to protect)")
    ap.add_argument("--leakage-gate", type=float, default=None,
                    help="exit 3 when the TEST-FACING strict pair count (train-test plus "
                         "valid-test) exceeds this. Use 0 to demand a clean test cohort. "
                         "Beds in GATE_EXEMPT_BEDS print their numbers and their reason "
                         "instead of failing -- see that dict for which and why.")
    ap.add_argument("--leakage-pairs-rebuild", action="store_true",
                    help="with --leakage-audit, ignore the cached strict pair list and "
                         "recompute it (a CPU image pass over the whole corpus)")
    ap.add_argument("--dedup-mask-rebuild", action="store_true",
                    help="with --dedup-mask, ignore any cached mask and recompute it")
    ap.add_argument("--no-dedup", action="store_true",
                    help="with --build, skip the deduplicated bed. Its first build decodes "
                         "the whole corpus on the CPU; afterwards the mask is cached.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pw-seed", type=int, default=42)
    ap.add_argument("--nick-valid-frac", type=float, default=0.15,
                    help="fraction of each class of the published training folder to carve "
                         "out as validation (default 0.15)")
    ap.add_argument("--require-nick", action="store_true",
                    help="fail if the published four-class corpus is not on disk, instead "
                         "of building the other beds without it")
    ap.add_argument("--content-group", choices=list(CONTENT_GROUP_MODES), default="md5",
                    help="how the validation carve groups training rows so that no group "
                         "spans the train/valid cut. 'md5' sees byte-identical copies "
                         "only; 'dihedral' sees a coarse quantized grid; 'strict' groups "
                         "by connected components of DEDUP_RULE_ID, the same relation the "
                         "deduplicator removes on, and is the only mode under which no "
                         "validation row is a strict duplicate of a training row. 'strict' "
                         "applies to the two nick beds; the non-reportable bal/imb carve "
                         "will raise rather than silently fall back.")
    ap.add_argument("--out-dir", default=SPLIT_DIR)
    a = ap.parse_args()

    if not any([a.build, a.show, a.check_caches, a.dup_audit, a.near_dup_audit,
                a.dedup_mask, a.leakage_audit]):
        ap.print_help()
        return 0

    beds: Dict[str, BedManifest] = {}
    if a.build:
        beds = build_all(seed=a.seed, pw_seed=a.pw_seed,
                         content_group=a.content_group, out_dir=a.out_dir,
                         nick_valid_frac=a.nick_valid_frac,
                         require_nick=a.require_nick,
                         build_dedup=not a.no_dedup, write=True)
        print(f"wrote {len(beds)} manifest(s) to {a.out_dir}\n")
        for k, bm in beds.items():
            why = non_reportable_reason(k)
            print(f"--- {k}" + ("   [NOT REPORTABLE]" if why else ""))
            print(bm.summary())
            if why:
                print(f"  not reportable: {why}")
            print()

    if a.show and not beds:
        for k in ALL_BEDS:
            try:
                bm = load_bed(k, a.out_dir)
            except FileNotFoundError as e:
                print(e)
                continue
            beds[k] = bm
            why = non_reportable_reason(k)
            print(f"--- {k}" + ("   [NOT REPORTABLE]" if why else ""))
            print(bm.summary())
            if why:
                print(f"  not reportable: {why}")
            print()

    if a.check_caches:
        if not beds:
            for k in ALL_BEDS:
                try:
                    beds[k] = load_bed(k, a.out_dir)
                except FileNotFoundError as e:
                    print(e)
        print("--- cache binding (headers only, no features loaded)")
        for k, bm in beds.items():
            idx = CacheIndex(cache_dirs_for(k))
            sizes = bed_source_sizes(bm)
            for bb in ("resnet18", "densenet121", "vit_b_16"):
                for role in ROLES:
                    m = bm.splits[role]
                    ok, note = True, ""
                    for sp in sorted(set(m.source_splits)):
                        try:
                            ref = idx.resolve(bb, _cache_split_key(sp), sizes[sp],
                                              tag=f"{k}/{bb}/{sp}")
                            note += f" {sp}->{os.path.basename(ref.path)[:46]}({ref.rows}x{ref.dim})"
                        except FileNotFoundError as e:
                            ok = False
                            note += f" {sp}->MISSING"
                            _ = e
                    print(f"  {'OK ' if ok else 'GAP'} {k:14s} {bb:12s} {role:5s} n={m.n:5d}{note}")
            if idx.ambiguous:
                for key, names in sorted(idx.ambiguous.items()):
                    print(f"      ambiguous {key}: {names}")

    if a.dup_audit:
        rep = content_duplicate_audit(
            roots={"bal": BED_ROOTS["bal"]}, class_names=FOUR_CLASSES,
            splits={"bal": ("Train", "Test")}, mode=a.dup_audit)
        os.makedirs(a.out_dir, exist_ok=True)
        p = os.path.join(a.out_dir, f"duplicate_audit_{a.dup_audit}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"--- duplicate audit ({a.dup_audit})")
        for kk in ("n_images", "n_distinct", "n_redundant_copies", "n_groups_with_copies",
                   "largest_group", "n_cross_split_groups", "n_cross_class_groups"):
            print(f"  {kk:24s} {rep[kk]}")
        print(f"  written to {p}")

    if a.dedup_mask:
        # Build (or reuse) the train-side duplicate mask on its own, so the expensive
        # part of the deduplicated bed can be inspected and argued with before any
        # manifest depends on it. Image files only: no features, no labels used for
        # the decision, no ledger read.
        root = BED_ROOTS["nick"]
        mp = dedup_mask_path("nick", a.out_dir)
        print(f"--- dedup mask for the four-class corpus at {root}")
        print(f"  rule {DEDUP_RULE_ID}")
        if not os.path.isdir(root):
            print(f"  no corpus there; nothing to do")
            return 2
        try:
            bm = build_deduplicated_four_class_bed(
                root, bed="nick_dedup", seed=a.seed,
                valid_frac=a.nick_valid_frac, content_group=a.content_group,
                mask_path=mp, write_mask=True, out_dir=a.out_dir,
                reuse_mask=not a.dedup_mask_rebuild, verbose=True)
        except RuntimeError as e:
            # The mask is written before these checks run, so it is still on disk
            # and still worth reporting even when it cannot support a bed.
            print(f"  mask built but unusable: {e}")
            return 4
        d = bm.audit["dedup"]
        print(f"  mask file      {mp}")
        print(f"  published pool {d['published_train_pool']} training rows")
        print(f"  removed        {d['removed']} ({100.0 * d['removed_fraction']:.1f}%)")
        print(f"  surviving      {d['surviving_train_pool']}")
        print(f"  per class      {d['per_class_removed']}")
        for kk in ("test", "leaked_train", "distinct_test_partners", "cross_label_pairs"):
            if kk in d["counts"]:
                print(f"  {kk:22s} {d['counts'][kk]}")
        print(f"\n  Nothing was removed from the test cohort: it stays byte-for-byte the "
              f"published one\n  ({bm.splits['test'].n} rows), so the standard and "
              f"deduplicated numbers share one cohort.\n  This run wrote no manifest; "
              f"use --build for that.")

    if a.leakage_audit:
        # The gate. Counted under DEDUP_RULE_ID -- the rule that removes rows -- so
        # that passing it is achievable by applying the remedy. The 0.98 audit below
        # stays a report: at 0.98 a bed cannot be required to be clean, because the
        # threshold also catches neighbouring slices of one patient, which are not
        # copies and which no deduplication should remove.
        bedk = a.leakage_bed
        root = BED_ROOTS[bedk]
        print(f"--- strict leakage audit ({bedk}) at {root}")
        print(f"  rule {DEDUP_RULE_ID}")
        if not os.path.isdir(root):
            print(f"  no corpus there; nothing to do")
            return 2
        try:
            bm = load_bed(bedk, a.out_dir)
        except FileNotFoundError as e:
            print(f"  {e}\n  This gate is about cross-ROLE pairs, and roles are what the "
                  f"manifest defines. Run --build first.")
            return 3

        folders = tuple(sorted({sp for role in ROLES
                                for sp in set(bm.splits[role].source_splits)}))
        os.makedirs(a.out_dir, exist_ok=True)
        rep = strict_leakage_audit(
            root, bm.splits, class_names=list(bm.class_names), folders=folders,
            out_dir=a.out_dir, bed=bedk,
            reuse=not a.leakage_pairs_rebuild, verbose=True)
        p = leakage_audit_path(bedk, a.out_dir)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)

        print(f"  folders            {list(folders)}")
        print(f"  images / pairs     {rep['n_images']} / {rep['n_strict_pairs']} "
              f"(manifest coverage {rep['manifest_coverage']})")
        print(f"  split sizes        {rep['split_sizes']}")
        print(f"  train-test         {rep['n_train_test']}")
        print(f"  valid-test         {rep['n_valid_test']}")
        print(f"  TEST-FACING        {rep['n_test_facing']}   <- what the gate reads")
        print(f"  train-valid        {rep['n_train_valid']}   "
              f"(reported, not gated; content_group='strict' is its fix)")
        print(f"  cross-class pairs  {rep['n_cross_class_pairs']} "
              f"(label noise, not leakage; {rep['n_test_facing_cross_class']} of them "
              f"test-facing)")
        print(f"  by role pair       {rep['role_pair_counts']}")
        if rep["rows_touched"]:
            print(f"  rows a fix touches {rep['rows_touched']} "
                  f"(non-test side only -- the test cohort is not ours to edit)")
        if "warning" in rep:
            print(f"  WARNING: {rep['warning']}")
        for key in ("test_train", "test_valid"):
            for r in rep["examples"].get(key, [])[:5]:
                print(f"    {key:11s} {r['corr']:.5f} {r['transform']:11s} "
                      f"resid={r['residual']:.2f}  {r['a']}  <->  {r['b']}")
        print(f"  {rep['caveat']}")
        print(f"  written to {p}")

        if a.leakage_gate is not None:
            n_bad = int(rep["n_test_facing"])
            limit = float(a.leakage_gate)
            if rep["gate_exempt"]:
                # Disclosure, not a pass. The number is printed at full volume
                # precisely because nothing will stop the run over it.
                print(f"\n  GATE EXEMPT: bed '{bedk}' is not gated.")
                print(f"  reason: {rep['gate_exempt_reason']}")
                print(f"  DISCLOSED: {n_bad} test-facing strict duplicate pair(s) "
                      f"({rep['n_train_test']} train-test, {rep['n_valid_test']} "
                      f"valid-test) involving "
                      f"{sum(rep['rows_touched'].get(k, 0) for k in ('test_train', 'test_valid'))}"
                      f" non-test rows. Any number measured on this bed carries that "
                      f"overlap and must be reported beside bed 'nick_dedup', never "
                      f"alone.")
            elif n_bad > limit:
                print(f"\n  GATE FAILED: {n_bad} strict duplicate pair(s) put a training "
                      f"or validation row in the test cohort (limit {limit:g}).")
                print(f"  by role pair: {rep['role_pair_counts']}")
                print(f"  These are pairs the deduplicator itself would remove, so this is "
                      f"not a threshold artifact: the reported number is measured partly "
                      f"on rows the model was fitted on. See {p} for the file list.")
                if bedk == "nick_dedup":
                    print(f"  For this bed the count is 0 by construction, so a non-zero "
                          f"one means the mask at {dedup_mask_path('nick', a.out_dir)} and "
                          f"the rule have drifted apart -- rebuild it with "
                          f"--dedup-mask --dedup-mask-rebuild.")
                return 3
            else:
                print(f"\n  gate passed: {n_bad} test-facing strict pair(s) <= {limit:g}"
                      + (" (0 by construction for this bed, and verified rather than "
                         "assumed)" if bedk == "nick_dedup" and n_bad == 0 else ""))
                if rep["n_train_valid"]:
                    print(f"  note: {rep['n_train_valid']} train-valid strict pair(s) "
                          f"remain. The test cohort is clean, but the validation score "
                          f"that picked the configuration is inflated by them; rebuild "
                          f"with --content-group strict to remove them.")

    if a.near_dup_audit:
        bedk = a.near_dup_bed
        root = BED_ROOTS[bedk]
        if bedk == "pw":
            # the patient-wise corpus already sits in train/valid/test folders
            cls = discover_classes(os.path.join(root, "train"))
            folders = ("train", "valid", "test")
        else:
            # Probe the folder names instead of assuming ("Train", "Test"): an
            # archive shipped as Training/ + Testing/ used to enumerate as zero
            # images here and still write a completed-looking report.
            cls = list(FOUR_CLASSES)
            folders = tuple(resolve_corpus_folder(root, r) for r in ("train", "test"))

        mans: Optional[Dict[str, SplitManifest]] = None
        try:
            bm = load_bed(bedk, a.out_dir)
            mans = bm.splits
            print(f"--- near-duplicate audit ({bedk}): manifests found, role-level "
                  f"counts included")
        except FileNotFoundError:
            print(f"--- near-duplicate audit ({bedk}): no manifest yet, folder-level only "
                  f"(run --build for role-level counts)")
            if a.near_dup_gate is not None:
                print("  refusing to gate on a folder-level count: --near-dup-gate is "
                      "about cross-ROLE pairs, which need the manifests. Run --build "
                      "first.")
                return 3

        os.makedirs(a.out_dir, exist_ok=True)
        rep = orientation_duplicate_audit(root, cls, splits=folders,
                                          side=a.near_dup_side,
                                          thresh=a.near_dup_thresh, manifests=mans)
        p = os.path.join(a.out_dir, f"near_duplicate_audit_{bedk}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        for kk in ("n_images", "n_pairs", "n_pairs_essentially_identical",
                   "n_pairs_needing_reorientation", "n_cross_folder_pairs",
                   "n_within_folder_pairs", "n_cross_class_pairs",
                   "manifest_coverage", "n_cross_role_pairs"):
            print(f"  {kk:32s} {rep[kk]}")
        if rep["role_pair_counts"]:
            print(f"  {'role_pair_counts':32s} {rep['role_pair_counts']}")
        for r in rep["cross_folder_pairs"][:10]:
            print(f"    cross-folder  {r['corr']:.4f} {r['transform']:11s} "
                  f"{r['a']}  <->  {r['b']}")
        for r in rep["cross_role_pairs"][:10]:
            print(f"    cross-role    {r['corr']:.4f} {r['transform']:11s} "
                  f"{r['a_role']}/{r['a']}  <->  {r['b_role']}/{r['b']}")
        for r in rep["cross_class_pairs"][:10]:
            print(f"    cross-class   {r['corr']:.4f} {r['transform']:11s} "
                  f"{r['a_class']}/{r['a']}  <->  {r['b_class']}/{r['b']}")
        print(f"  {rep['caveat']}")
        print(f"  written to {p}")

        if a.near_dup_sweep:
            sw = orientation_threshold_sweep(root, cls, splits=folders,
                                            side=a.near_dup_side, manifests=mans)
            q = os.path.join(a.out_dir, f"near_duplicate_sweep_{bedk}.json")
            with open(q, "w", encoding="utf-8") as f:
                json.dump(sw, f, ensure_ascii=False, indent=2)
            print(f"--- threshold sweep ({bedk})")
            print(f"  {'thresh':>8s} {'pairs':>8s} {'x-folder':>9s} {'x-role':>8s} "
                  f"{'x-class':>8s} {'reoriented':>11s}")
            for r in sw["rows"]:
                xr = "-" if r["n_cross_role"] is None else r["n_cross_role"]
                print(f"  {r['threshold']:8.3f} {r['n_pairs']:8d} {r['n_cross_folder']:9d} "
                      f"{str(xr):>8s} {r['n_cross_class']:8d} {r['n_reoriented']:11d}")
            print(f"  written to {q}")

        # The audit is only worth running if it can stop something. Without a gate
        # it writes a JSON nobody is obliged to open, which is how a 72%-leaky bed
        # stayed reportable for two revisions.
        if a.near_dup_gate is not None:
            n_bad = int(rep["n_cross_role_pairs"])
            limit = float(a.near_dup_gate)
            role_pairs = rep["role_pair_counts"]
            if n_bad > limit:
                print(f"\n  GATE FAILED: {n_bad} near-duplicate pair(s) cross a train/valid/"
                      f"test boundary at threshold {a.near_dup_thresh} (limit {limit:g}).")
                print(f"  by role pair: {role_pairs}")
                print(f"  A train-test or valid-test pair contaminates the reported number; "
                      f"a train-valid pair inflates the validation score that fixed the "
                      f"thresholds and picked the configuration. See {p} for the file list.")
                return 3
            print(f"\n  gate passed: {n_bad} cross-role pair(s) <= {limit:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
