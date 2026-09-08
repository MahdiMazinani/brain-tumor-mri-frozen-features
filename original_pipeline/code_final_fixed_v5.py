"""code_final_fixed_v5.py -- the frozen-embedding hybrid pipeline, rebuilt so that
the two grounds of the JMI desk rejection are answered by construction rather than
by assertion.

What v4 did and why it was not enough
-------------------------------------
v4 produced correct numbers for a *fixed* configuration, but the surrounding
protocol could not support the claims made around it.

  (1) "does not yet establish how the proposed method compares to the most
      relevant published work under directly comparable conditions: the same
      cohorts, training and validation splits, preprocessing, evaluation criteria,
      and held-out test cases."

      v4 had no notion of a competitor. Reported baselines were literature numbers
      obtained on other splits. v5 makes a competitor a first-class object: every
      entry in BASELINES is a frozen-feature recipe that runs through the *same*
      manifest, the *same* preprocessing, the *same* decision rule and the *same*
      sealed test cohort as the proposed method, and lands in the same results
      table. A reader can then compare rows, not papers.

  (2) "the final test cohort must also stay fully independent of method
      development, not used for training, validation, model or hyperparameter
      selection, or threshold optimization, since such use can inflate reported
      performance."

      v4 loaded train, valid and test in one call and handed all three to every
      learner, so per-class thresholds, ensemble weights and the choice among 35
      configurations were all computed with test probabilities already in memory.
      Nothing enforced the ordering. v5 splits the run into a selection phase that
      can only see validation and a reporting phase that must call
      SelectionLedger.authorize_test_read() -- which raises before freeze() and
      counts reads after it.

Three further defects fixed here
--------------------------------
  * v4's validation split came from sklearn train_test_split on image rows, which
    is leaky for grouped data and produced a different split on every run. v5
    consumes the hashed manifests built by splits_v5.py; the validation cohort is
    a deterministic, group-respecting carve-out of the training folder.
  * v4's confidence interval resampled slices i.i.d., which is wrong when several
    slices come from one patient. v5 defaults to a patient-clustered bootstrap on
    any bed that has identifiers, and reports the i.i.d. interval beside it only
    as a demonstration of how much narrower it wrongly looks.
  * v4 keyed its feature cache on per-class file counts *and mtimes*. A bulk copy
    rewrote the mtimes, so none of the 54 cached .npy files on disk can be
    re-derived from their key any more. v5 keys new caches on the manifest digest
    and binds pre-existing caches by (backbone, split, row count) with an
    assertion, never by filesystem metadata.

The learner stage is deliberately *not* reimplemented. code_final_fixed_v4.py is
imported and its train_*_proba, threshold, ensemble and metric functions are
called directly, so the classifiers here are the same code that produced the
submitted numbers. Only the protocol around them changed.

Usage (nothing runs on import):

    python splits_v5.py --build                     # once: write the manifests
    python code_final_fixed_v5.py --bed bal --dry-run
    python code_final_fixed_v5.py --bed bal --stage select
    python code_final_fixed_v5.py --bed bal --stage report
    python code_final_fixed_v5.py --bed pw_family --stage cv --folds 5
    python code_final_fixed_v5.py --bed bal --stage baselines
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import splits_v5 as S  # noqa: E402
import preprocess_v5 as PRE  # noqa: E402

BASE_CODE = os.path.join(HERE, "code_final_fixed_v4.py")


def _load_base():
    """Import code_final_fixed_v4 as the reference implementation.

    Deliberately a lazy function rather than a module-level import: splits can be
    built, inspected and unit-checked on a machine with no torch installed, and
    only the stages that actually train a classifier need the heavy dependency.
    """
    if not os.path.exists(BASE_CODE):
        raise FileNotFoundError(
            f"{BASE_CODE} is missing. v5 does not reimplement the learners; it calls "
            f"v4's train_*_proba / tune_thresholds / eval_metrics_general so that the "
            f"classifier code behind the reported numbers stays byte-identical.")
    spec = importlib.util.spec_from_file_location("base_v4", BASE_CODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_B = None


def B():
    global _B
    if _B is None:
        _B = _load_base()
    return _B

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKBONES = ("resnet18", "densenet121", "vit_b_16")
BACKBONE_DIM = {"resnet18": 512, "densenet121": 1024, "vit_b_16": 768}
SHORT = {"resnet18": "R18", "densenet121": "D121", "vit_b_16": "ViT"}

SPECS7 = (
    "resnet18",
    "densenet121",
    "vit_b_16",
    "resnet18+densenet121",
    "resnet18+vit_b_16",
    "densenet121+vit_b_16",
    "resnet18+densenet121+vit_b_16",
)

LEARNERS = ("svm", "xgb", "lgb", "cat")
LEARNER_LABEL = {"svm": "SVM-RBF", "xgb": "XGBoost", "lgb": "LightGBM",
                 "cat": "CatBoost", "ens": "Ensemble"}


def spec_label(spec: str) -> str:
    return "+".join(SHORT[b] for b in spec.split("+"))


@dataclass
class Cfg5:
    """Everything a run needs. Fields are grouped by who is allowed to set them.

    The `selectable_*` fields are the ones the ledger tracks: they may only be
    changed on the strength of validation evidence, before freeze(). The rest are
    fixed by the protocol and are recorded so a rerun is reproducible.
    """
    bed: str = "bal"
    split_dir: str = ""
    run_dir: str = ""
    img_size: int = 224

    # protocol, not selectable
    random_state: int = 42
    normal_class: str = "auto"
    patient_level: bool = True
    bootstrap_n: int = 2000
    cluster_bootstrap: bool = True
    report_iid_bootstrap_too: bool = True
    threshold_metric: str = "bal_acc"
    min_tumor_recall: float = 0.0

    # selectable on validation only
    specs: Tuple[str, ...] = SPECS7
    learners: Tuple[str, ...] = LEARNERS
    pca_var: float = 0.95
    standardize: str = "per_dim"
    svm_c: float = 1.5
    xgb_rounds: int = 1200
    lgb_rounds: int = 1000
    cat_rounds: int = 1000
    tta: int = 3
    decision_rule: str = "auto"          # argmax | threshold | bayes | bayes_cal | auto
    use_hard_weighting: bool = True
    hard_weight_gamma: float = 2.0
    hard_weight_fn_penalty: float = 5.0
    optimize_ensemble: bool = True
    calibrate: bool = True

    # cost matrix of the minimum-expected-risk rule
    c_miss: float = 10.0
    c_type: float = 4.0
    c_fp: float = 1.0

    # execution
    xgb_gpu: bool = True
    lgbm_try_gpu: bool = False
    prefer_existing_caches: bool = True
    # `prefer_existing_caches=False` re-extracts *everything*, including the three
    # backbones whose caches produced the submitted numbers. That is almost never
    # what is wanted. `extract_missing` is the narrow version: bind to a cache
    # wherever one exists, and extract only the backbones that have none -- which
    # is the whole competitor set, since no googlenet/vgg19/densenet169/resnext50
    # features were ever extracted. Off by default because it needs a GPU.
    extract_missing: bool = False
    strict_test_seal: bool = True
    # Legacy caches do not record their TTA setting in the filename, so it is inferred
    # from row norm (see splits_v5.CacheIndex.classify). Where the requested setting was
    # never extracted at all -- tta=3 validation on the four-class beds, whose rows come
    # out of the never-augmented Train cache -- there are two honest options: serve what
    # exists and say so in the bind log, or refuse. Default is to serve and record, which
    # keeps the ablation runnable; ABLATION_STRICT_TTA=1 or --strict-tta flips it to
    # refuse, which is the right setting when a table's whole claim is the TTA contrast.
    strict_tta: bool = S.STRICT_TTA_DEFAULT
    class_names: Tuple[str, ...] = ()

    # reporting
    refit_on_train_valid: bool = False
    baselines: Tuple[str, ...] = ()      # empty means every registered baseline
    baseline_tta: int = 1
    # A same-condition table that quietly drops the competitors it could not run does
    # not answer the comparability question -- it looks like an answer. So the
    # baselines stage refuses to write one unless this is set deliberately.
    allow_partial_baselines: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    def digest(self) -> str:
        return hashlib.md5(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()

# ---------------------------------------------------------------------------
# Bed resolution
# ---------------------------------------------------------------------------
# A "bed" is a manifest plus the metadata the metrics need. The four keys are
# bal, imb, pw_identifier and pw_family; "pw" is accepted as an alias for the
# identifier variant, because that is the split the submitted manuscript used.

_NORMAL_ALIASES = ("notumor", "no_tumor", "normal", "healthy", "nontumor", "no tumor")


def normal_index(class_names: Sequence[str], requested: str = "auto") -> Optional[int]:
    names = [str(c).strip().lower().replace("-", "_").replace(" ", "_") for c in class_names]
    if requested and requested != "auto":
        r = requested.strip().lower().replace("-", "_").replace(" ", "_")
        return names.index(r) if r in names else None
    for a in _NORMAL_ALIASES:
        a = a.replace(" ", "_")
        if a in names:
            return names.index(a)
    return None


@dataclass
class Bed5:
    key: str
    manifest: "S.BedManifest"
    class_names: Tuple[str, ...]
    normal_idx: Optional[int]
    group_unit: str
    has_pids: bool

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def split(self, role: str) -> "S.SplitManifest":
        return self.manifest.splits[role]

    def groups(self, role: str) -> np.ndarray:
        return np.asarray(self.manifest.splits[role].groups)

    def cost_matrix(self, c_miss: float, c_type: float, c_fp: float) -> np.ndarray:
        """Cost of predicting column j when the truth is row i.

        Asymmetric on purpose: calling a tumour healthy is the failure with
        clinical consequence, calling a healthy scan tumoral costs a follow-up, and
        naming the wrong tumour type still flags the lesion. The manuscript names
        this rule as the principled fix for thresholding but never evaluated it;
        v5 evaluates it and can select it on validation.
        """
        K = self.n_classes
        M = np.zeros((K, K), dtype=np.float64)
        n = self.normal_idx
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                if n is not None and i != n and j == n:
                    M[i, j] = c_miss
                elif n is not None and i == n and j != n:
                    M[i, j] = c_fp
                else:
                    M[i, j] = c_type
        return M


def load_bed(bed_key: str, split_dir: Optional[str] = None,
             normal_class: str = "auto") -> Bed5:
    bm = S.load_bed(bed_key, split_dir)
    cls = tuple(bm.class_names)
    has_pids = any(S.pid_from_path(f) != os.path.splitext(os.path.basename(f))[0]
                   for f in bm.splits["test"].files[:50])
    return Bed5(key=bed_key, manifest=bm, class_names=cls,
                normal_idx=normal_index(cls, normal_class),
                group_unit=bm.group_unit, has_pids=has_pids)

# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
# Two paths, and the distinction matters for cost.
#
# Fast path: every row of a manifest names the folder and row index where its
# feature vector already sits, so a 1,600-row validation cohort carved out of the
# 5,600-row train cache is a gather, not 1,600 forward passes. This is why the
# four-class beds could be rebuilt without a GPU: all 9,200 vectors of bal and imb
# are rows of caches extracted in August.
#
# Slow path: if a manifest asks for something no cache covers -- a new TTA setting,
# a new backbone, a new image size -- extract_manifest() runs the backbone over the
# manifest's file list *in manifest order* and writes a cache keyed on the manifest
# digest. Keying on the digest, not on file mtimes, is the lesson of v4: mtimes are
# rewritten by any copy, and once they change the key can never be reproduced.

class FeatureStore:
    """Materialize (X, y) for a manifest, from cache when possible.

    `tta_n` is part of the identity of a feature matrix, not a detail: TTA averages
    embeddings over transformed views, so the same images at tta=1 and tta=3 are
    different vectors. v4's caches encode this only through the dead fingerprint,
    which is why both test caches in each bed have identical shapes and cannot be
    told apart by name. `prefer` lets a caller pin a specific file, and every
    resolution is written into the run artifact so the choice is on record.
    """

    def __init__(self, cfg: Cfg5, bed: Bed5):
        self.cfg = cfg
        self.bed = bed
        # The bed key goes to cache_dirs_for unmapped: it applies the cache-family
        # mapping itself (imb, nick and nick_dedup all read the bal caches, since
        # all three are row selections over one extraction), and mapping here too
        # would be a second copy of that rule to keep in step. Passing the family
        # instead used to be harmless only because family and cache directory
        # happened to coincide for every bed that existed.
        self.index = S.CacheIndex(S.cache_dirs_for(bed.key) + [self._own_cache_dir()],
                                 img_size=cfg.img_size,
                                 strict_tta=bool(cfg.strict_tta))
        self.source_sizes = S.bed_source_sizes(bed.manifest)
        self._mem: Dict[Tuple[str, str, str, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.binds: List[Dict[str, Any]] = []

    def _own_cache_dir(self) -> str:
        d = os.path.join(S.CACHE_ROOT, f"cache_{self.bed.key}_v5_s{self.cfg.img_size}")
        os.makedirs(d, exist_ok=True)
        return d

    # -- v5-native cache, keyed on the manifest digest ---------------------
    def _v5_paths(self, m: "S.SplitManifest", backbone: str, tta_n: int) -> Tuple[str, str]:
        # The preprocessing rule id joins the key for the same reason img_size and tta
        # are already in it: it is a thing that changes the pixels, so a cache written
        # under one value must not be served under another. Safe to add now because no
        # v5-native cache exists to invalidate: _own_cache_dir() creates
        # cache_<bed>_v5_s224 lazily on the first miss and, measured 2026-09-06, the
        # only directories under ablation/caches are the two legacy v4 ones.
        # The legacy v4 caches are resolved by CacheIndex on row count and are not
        # keyed by this, which is why the pin is verified against v4's transform rather
        # than merely declared: those caches produced the submitted numbers.
        key = hashlib.md5(
            f"{m.digest()}|{backbone}|s{self.cfg.img_size}|tta{int(tta_n)}"
            f"|{PRE.preprocess_rule_id(self.cfg.img_size, backbone)}".encode()
        ).hexdigest()
        d = self._own_cache_dir()
        tag = f"{backbone}_s{self.cfg.img_size}"
        return (os.path.join(d, f"X_{tag}_{m.role}_{key}.npy"),
                os.path.join(d, f"y_{tag}_{m.role}_{key}.npy"))

    def backbone_features(self, role: str, backbone: str,
                          tta_n: int) -> Tuple[np.ndarray, np.ndarray]:
        m = self.bed.split(role)
        ck = (self.bed.key, role, backbone, int(tta_n))
        if ck in self._mem:
            return self._mem[ck]

        xp, yp = self._v5_paths(m, backbone, tta_n)
        if os.path.exists(xp) and os.path.exists(yp):
            X, y = np.load(xp).astype(np.float32), np.load(yp).astype(np.int64)
            # The v5-native path cannot substitute: tta is inside the cache key, so a
            # hit is by definition the requested setting. Recorded in the same two
            # fields as the legacy path so a reader of the bind log never has to know
            # which path produced a row in order to know what TTA it holds.
            self.binds.append(dict(role=role, backbone=backbone,
                                   tta_requested=int(tta_n), tta_effective=int(tta_n),
                                   tta_substituted=False, tta_note="",
                                   how="v5_manifest_cache", file=os.path.basename(xp),
                                   rows=int(X.shape[0]), dim=int(X.shape[1]),
                                   label_check="cache_y_vs_manifest_y"))
        elif self.cfg.prefer_existing_caches:
            try:
                X, y = self._from_existing(m, backbone, tta_n)
            except FileNotFoundError:
                # No cache exists for this backbone at all. That is the normal state
                # for the competitor backbones (googlenet, vgg19, densenet169,
                # resnext50): only the three of the proposed method were ever
                # extracted. Extracting just those, and only on request, keeps the
                # proposed method's numbers bound to the caches that produced them
                # while still letting the same-condition table be completed.
                if not self.cfg.extract_missing:
                    raise
                print(f"  [extract] no cache for {backbone}/{role}; extracting "
                      f"{m.n} rows at tta={tta_n} (needs a GPU to be quick)",
                      flush=True)
                X, y = self.extract(role, backbone, tta_n)
        else:
            X, y = self.extract(role, backbone, tta_n)

        if X.shape[0] != m.n:
            raise RuntimeError(f"{self.bed.key}/{role}/{backbone}: got {X.shape[0]} feature "
                               f"rows for a {m.n}-row manifest")
        if not np.array_equal(y, m.y()):
            raise RuntimeError(
                f"{self.bed.key}/{role}/{backbone}: cached labels disagree with the manifest. "
                f"Either the cache belongs to a different split or the class order changed. "
                f"Refusing to train on it.")
        self._mem[ck] = (X, y)
        return X, y

    def _from_existing(self, m: "S.SplitManifest", backbone: str,
                       tta_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Bind to the caches extracted before v5 existed.

        Four things are asserted, in increasing strength. The row count of the cache
        must equal the row count of the source folder, or resolve raises. The file must
        hold the TTA setting being asked for -- decided by row norm, since the filename
        does not record it -- or the substitution is named and recorded. The gathered
        rows must fill the manifest exactly, or assemble_features raises. And the labels
        stored beside the features, which encode the class order in force at extraction
        time, must match the manifest's labels, or this raises. Without the last check a
        renamed or reordered class folder would produce a permuted target that trains
        happily and reports nonsense.

        The TTA assertion is why `tta_n` is threaded down instead of only being logged.
        Every four-class bed holds two shape-identical test caches per backbone, one per
        setting; binding by row count alone picks whichever is newer, so a tta=1 request
        could be served the averaged file and the bind log would still say tta=1. The
        same-condition table turns on exactly that distinction -- baseline_tta=1 against
        a proposed row at tta=3 has to be a real difference, not a nominal one.
        """
        prefer = None
        X, y = S.assemble_features(m, self.index, backbone,
                                   source_rows_total=self.source_sizes, prefer=prefer,
                                   tta=int(tta_n))
        y_cached = S.assemble_labels(m, self.index, backbone,
                                     source_rows_total=self.source_sizes, prefer=prefer,
                                     tta=int(tta_n))
        if y_cached is None:
            check = "NO_CACHED_LABELS_row_count_only"
            warnings.warn(
                f"{self.bed.key}/{m.role}/{backbone}: the resolved cache has no y_*.npy "
                f"beside it, so the bind rests on row counts alone. Verified: the cache "
                f"covers the source folder in full and the manifest indexes into it "
                f"without going out of range. Not verified: that the class order at "
                f"extraction time matches the current folder order.")
        elif not np.array_equal(y_cached, m.y()):
            n_bad = int((np.asarray(y_cached) != m.y()).sum())
            raise RuntimeError(
                f"{self.bed.key}/{m.role}/{backbone}: {n_bad} of {m.n} labels stored with "
                f"the cached features disagree with the manifest. The cache was extracted "
                f"under a different class order than {list(self.bed.class_names)}. "
                f"Re-extract with --extract rather than training on a permuted target.")
        else:
            check = "cache_y_vs_manifest_y"
        for sp in sorted(set(m.source_splits)):
            # Same key the resolver filed the bind under -- S._bind_tag, not a second
            # spelling of it. A miss here would silently fall back to `tta_n`, the
            # requested value, and the log would once again assert a condition it had
            # not checked.
            tag = S._bind_tag(m, backbone, sp)
            ref = self.index.resolved.get(tag)
            tb = self.index.tta_binds.get(tag, {})
            if not tb:
                raise RuntimeError(
                    f"{self.bed.key}/{m.role}/{backbone}: no TTA bind was recorded for "
                    f"source folder {sp!r} under key {tag!r}, so the effective setting is "
                    f"unknown. Present keys: {sorted(self.index.tta_binds)}. Refusing to "
                    f"write a bind log that would report tta={tta_n} without having "
                    f"verified it.")
            # requested and effective are recorded separately, always. A bind log that
            # prints the requested value as if it were honoured is the same defect the
            # desk rejection objected to -- a reported condition the run did not meet --
            # one level down.
            self.binds.append(dict(role=m.role, backbone=backbone,
                                   tta_requested=int(tb["requested"]),
                                   tta_effective=int(tb["effective"]),
                                   tta_substituted=bool(tb.get("substituted", False)),
                                   tta_note=tb.get("note", ""),
                                   tta_classification=tb.get("classification", ""),
                                   how="legacy_cache_by_rowcount_and_tta", source_split=sp,
                                   file=os.path.basename(ref.path) if ref else "?",
                                   rows=ref.rows if ref else -1,
                                   source_folder_rows=int(self.source_sizes.get(sp, -1)),
                                   label_check=check,
                                   ambiguous=self.index.ambiguous.get(tag)))
        return X, y

    # -- the slow path -----------------------------------------------------
    def extract(self, role: str, backbone: str, tta_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run the frozen backbone over a manifest, in manifest order.

        Uses v4's build_extractor / tta_variants and preprocess_v5's pinned transform,
        so the embeddings are the same function of the pixels as before -- the pin is
        verified equal to v4's transform, not assumed to be. Only the row order, the
        cache key and the explicitness of the resize come from v5.
        """
        import torch
        from PIL import Image

        base = B()
        m = self.bed.split(role)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base.set_torch_fast()
        extractor, _dim = build_extractor_v5(backbone, device, self.cfg.img_size)
        # The pinned pixel path instead of v4's defaulted one. Identical output --
        # PRE.assert_matches_reference() checks that with torch.equal on four input
        # geometries, and --stage dry-run runs it -- but with every argument written
        # down, so a torchvision upgrade cannot move the embeddings underneath a
        # cached number. norm_for() also raises here for an unregistered backbone,
        # which is better than normalizing a competitor with constants that are not
        # its own and reporting the result in a same-condition table.
        pre = PRE.build_transform(self.cfg.img_size, backbone)
        variants = base.tta_variants(int(tta_n))
        is_vit = backbone.lower().strip() in ("vit_b_16", "vit", "vitb16")

        paths = m.abs_paths()
        bs = 16
        chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for i0 in range(0, len(paths), bs):
                imgs = [Image.open(p).convert("RGB") for p in paths[i0:i0 + bs]]
                acc = None
                for _, vf in variants:
                    xb = torch.stack([pre(vf(im)) for im in imgs], dim=0).to(device)
                    if device.type == "cuda" and not is_vit:
                        xb = xb.to(memory_format=torch.channels_last)
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            f = extractor(xb)
                    else:
                        f = extractor(xb)
                    if f.dim() == 4:
                        f = f.squeeze(-1).squeeze(-1)
                    f = f.detach().float()
                    acc = f if acc is None else acc + f
                chunks.append((acc / float(len(variants))).cpu().numpy().astype(np.float32))
                for im in imgs:
                    im.close()
        X = np.concatenate(chunks, axis=0)
        y = m.y()
        xp, yp = self._v5_paths(m, backbone, tta_n)
        np.save(xp, X)
        np.save(yp, y)
        self.binds.append(dict(role=role, backbone=backbone,
                               tta_requested=int(tta_n), tta_effective=int(tta_n),
                               tta_substituted=False, tta_note="",
                               how="extracted_now", file=os.path.basename(xp),
                               rows=int(X.shape[0]), dim=int(X.shape[1])))
        del extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return X, y

    def spec_features(self, role: str, spec: str, tta_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate the raw embeddings of every backbone in a fusion spec."""
        Xs, y_ref = [], None
        for bb in spec.split("+"):
            X, y = self.backbone_features(role, bb, tta_n)
            if y_ref is None:
                y_ref = y
            elif not np.array_equal(y, y_ref):
                raise RuntimeError(f"label mismatch fusing {spec} on {role}")
            Xs.append(X)
        X = Xs[0] if len(Xs) == 1 else np.concatenate(Xs, axis=1)
        return X, y_ref

    def effective_tta(self, role: str, spec: Optional[str] = None,
                      requested: Optional[int] = None) -> Dict[str, Any]:
        """What TTA setting the matrices of `role` were actually built at.

        Reads the bind log rather than the config, which is the whole point: cfg.tta is
        a request. Returned as a dict so a caller can write both numbers into the
        ledger, and so a mixed answer (one backbone substituted, another not) is
        visible instead of being collapsed to a single integer.

        `spec` and `requested` narrow the answer to one fusion spec and one request,
        which matters in the baselines stage: binds accumulate across competitors, so an
        unfiltered query would mix one competitor's substitution into another's row.
        """
        want_bb = set(spec.split("+")) if spec else None
        rows = [b for b in self.binds
                if b.get("role") == role
                and (want_bb is None or b.get("backbone") in want_bb)
                and (requested is None or int(b.get("tta_requested", -1)) == int(requested))]
        eff = sorted({int(b["tta_effective"]) for b in rows if "tta_effective" in b})
        req = sorted({int(b["tta_requested"]) for b in rows if "tta_requested" in b})
        subs = sorted({(b["backbone"], int(b["tta_requested"]), int(b["tta_effective"]))
                       for b in rows if b.get("tta_substituted")})
        return dict(requested=req, effective=eff, substituted=subs,
                    uniform=len(eff) == 1,
                    notes=sorted({b["tta_note"] for b in rows if b.get("tta_note")}))

    def effective_tta_scalar(self, role: str, spec: Optional[str] = None,
                             requested: Optional[int] = None) -> Any:
        """effective_tta collapsed to an int when unambiguous, else the list."""
        info = self.effective_tta(role, spec=spec, requested=requested)
        if info["uniform"] and info["effective"]:
            return int(info["effective"][0])
        return info["effective"]

    def report(self) -> Dict[str, Any]:
        return dict(binds=self.binds, cache_index=self.index.report(),
                    effective_tta={r: self.effective_tta(r) for r in S.ROLES},
                    source_split_sizes=self.source_sizes)


def spec_dims(spec: str) -> List[int]:
    # ALL_DIM is defined further down (it adds the competitor backbones to
    # BACKBONE_DIM); module-level lookup at call time picks it up.
    return [ALL_DIM[b] for b in spec.split("+")]

# ---------------------------------------------------------------------------
# Representation stage
# ---------------------------------------------------------------------------
# Fitted on TRAIN rows only. This was already true in v4 and is kept, but v5 makes
# it structural: fit_representation never receives the test matrix at all. The test
# matrix is transformed later, by a Representation object, and only after the
# ledger has authorized the read.

@dataclass
class Representation:
    scaler: Any = None
    pca: Any = None
    mode: str = "per_dim"
    block_stats: Optional[List[Tuple[float, float]]] = None
    dims: Optional[List[int]] = None
    n_components: int = 0
    explained: float = 0.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=np.float32)
        if self.mode == "per_dim" and self.scaler is not None:
            Z = self.scaler.transform(Z)
        elif self.mode == "per_bb" and self.block_stats is not None:
            Z = Z.copy()
            off = 0
            for (mu, sd), d in zip(self.block_stats, self.dims or []):
                Z[:, off:off + d] = (Z[:, off:off + d] - mu) / sd
                off += d
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return np.asarray(Z, dtype=np.float32)


def fit_representation(X_tr: np.ndarray, mode: str, pca_var: float,
                       dims: Optional[Sequence[int]] = None,
                       seed: int = 42) -> Tuple[Representation, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rep = Representation(mode=mode, dims=list(dims) if dims else None)
    Z = np.asarray(X_tr, dtype=np.float32)
    if mode == "per_dim":
        rep.scaler = StandardScaler()
        Z = rep.scaler.fit_transform(Z)
    elif mode == "per_bb":
        if not dims:
            raise ValueError("per_bb standardization needs the per-backbone dims")
        Z = Z.copy()
        stats, off = [], 0
        for d in dims:
            blk = Z[:, off:off + d]
            mu = float(blk.mean())
            sd = float(blk.std()) or 1.0
            Z[:, off:off + d] = (blk - mu) / sd
            stats.append((mu, sd))
            off += d
        rep.block_stats = stats
    elif mode != "none":
        raise ValueError(f"unknown standardization mode {mode!r}")

    if pca_var and pca_var > 0:
        rep.pca = PCA(n_components=float(pca_var), random_state=seed)
        Z = rep.pca.fit_transform(Z)
        rep.n_components = int(rep.pca.n_components_)
        rep.explained = float(np.sum(rep.pca.explained_variance_ratio_))
    else:
        rep.n_components = int(Z.shape[1])
        rep.explained = 1.0
    return rep, np.asarray(Z, dtype=np.float32)

# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
# Four rules on identical posteriors. The manuscript reported per-class
# thresholding as a negative result and diagnosed the cause correctly -- dividing
# each posterior by an independently tuned threshold breaks the simplex, so the
# compared quantities are no longer commensurable -- but then proposed the fix
# (calibration plus a cost-matrix Bayes rule) without testing it. Here the fix is
# implemented, and which rule is used is a validation decision recorded in the
# ledger, never a test-set decision.

def temperature_fit(proba_va: np.ndarray, y_va: np.ndarray) -> float:
    """One scalar T minimizing validation NLL of softmax(log p / T)."""
    from scipy.optimize import minimize_scalar
    logp = np.log(np.clip(proba_va, 1e-12, 1.0))
    yv = np.asarray(y_va)

    def nll(logT: float) -> float:
        T = float(np.exp(logT))
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return -float(np.mean(np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1.0))))

    r = minimize_scalar(nll, bounds=(float(np.log(0.05)), float(np.log(20.0))),
                        method="bounded")
    return float(np.exp(r.x))


def temperature_apply(proba: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-12, 1.0)) / float(T)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def bayes_decision(proba: np.ndarray, M: np.ndarray) -> np.ndarray:
    """argmin_j sum_i p(i|x) C[i,j] -- minimum expected risk."""
    return np.asarray(proba @ M).argmin(axis=1)


def expected_risk(y_true: np.ndarray, y_pred: np.ndarray, M: np.ndarray) -> float:
    return float(np.mean([M[int(a), int(b)] for a, b in zip(y_true, y_pred)]))


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    ok = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            out += float(m.mean()) * abs(float(ok[m].mean()) - float(conf[m].mean()))
    return float(out)


def apply_rule(proba: np.ndarray, rule: str, *, thresholds: Optional[np.ndarray] = None,
               normal_idx: Optional[int] = None, cost: Optional[np.ndarray] = None,
               temperature: Optional[float] = None) -> np.ndarray:
    if rule == "argmax":
        return proba.argmax(axis=1)
    if rule == "threshold":
        if thresholds is None:
            raise ValueError("the thresholded rule needs thresholds tuned on validation")
        return B().predict_with_thresholds(proba, np.asarray(thresholds), normal_idx)
    if rule in ("bayes", "bayes_cal"):
        if cost is None:
            raise ValueError("a Bayes rule needs a cost matrix")
        p = proba if rule == "bayes" else temperature_apply(proba, float(temperature or 1.0))
        return bayes_decision(p, cost)
    raise ValueError(f"unknown decision rule {rule!r}")

# ---------------------------------------------------------------------------
# Metrics and resampling
# ---------------------------------------------------------------------------

def fnr_presence(y_true: np.ndarray, y_pred: np.ndarray,
                 normal_idx: Optional[int]) -> Optional[float]:
    """Share of tumour cases predicted healthy -- the error with consequence."""
    if normal_idx is None:
        return None
    m = np.asarray(y_true) != normal_idx
    if not m.any():
        return None
    return float((np.asarray(y_pred)[m] == normal_idx).mean())


def type_error_rate(y_true: np.ndarray, y_pred: np.ndarray,
                    normal_idx: Optional[int]) -> Optional[float]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    m = (yt != normal_idx) if normal_idx is not None else np.ones(len(yt), bool)
    if not m.any():
        return None
    wrong = yp[m] != yt[m]
    if normal_idx is not None:
        wrong &= yp[m] != normal_idx
    return float(wrong.mean())


def score_block(bed: Bed5, y_true: np.ndarray, y_pred: np.ndarray,
                prefix: str = "") -> Dict[str, float]:
    base = B()
    m = base.eval_metrics_general(np.asarray(y_true), np.asarray(y_pred),
                                  list(bed.class_names))
    out = {f"{prefix}acc": m["accuracy"], f"{prefix}bal_acc": m["balanced_accuracy"],
           f"{prefix}f1": m["f1_macro"], f"{prefix}prec": m["precision_macro"],
           f"{prefix}rec": m["recall_macro"], f"{prefix}fnr_macro": m["FNR_macro"]}
    fp = fnr_presence(y_true, y_pred, bed.normal_idx)
    if fp is not None:
        out[f"{prefix}fnr_pres"] = fp
    tp = type_error_rate(y_true, y_pred, bed.normal_idx)
    if tp is not None:
        out[f"{prefix}type_err"] = tp
    for c in bed.class_names:
        out[f"{prefix}rec_{c}"] = m[f"recall_{base.safe_name(c)}"]
        out[f"{prefix}prec_{c}"] = m[f"precision_{base.safe_name(c)}"]
    return out


def cluster_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                      n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Resample *patients* with replacement, then score all their slices.

    This is the correction v4 needed. When 781 test slices come from 58 patients,
    an i.i.d. slice bootstrap treats correlated observations as independent and
    reports an interval that is too narrow -- the effective sample size is nearer 58
    than 781. Every interval in the manuscript's patient-wise bed should be this
    one; the i.i.d. version is reported alongside only to show the size of the error.
    """
    rng = np.random.default_rng(seed)
    yt, yp, g = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    uniq = np.unique(g)
    idx_by_g = {u: np.flatnonzero(g == u) for u in uniq}
    accs = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[u] for u in drawn])
        accs[b] = float((yt[idx] == yp[idx]).mean())
    return dict(mean=float(accs.mean()), lo=float(np.percentile(accs, 2.5)),
                hi=float(np.percentile(accs, 97.5)), unit="patient",
                n_units=int(len(uniq)), n_boot=int(n_boot))


def iid_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000,
                  seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    accs = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        i = rng.integers(0, n, n)
        accs[b] = float((yt[i] == yp[i]).mean())
    return dict(mean=float(accs.mean()), lo=float(np.percentile(accs, 2.5)),
                hi=float(np.percentile(accs, 97.5)), unit="slice",
                n_units=int(n), n_boot=int(n_boot))


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Exact two-sided binomial McNemar on the discordant pairs."""
    yt = np.asarray(y_true)
    a_ok, b_ok = (np.asarray(pred_a) == yt), (np.asarray(pred_b) == yt)
    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    n = n01 + n10
    if n == 0:
        return dict(n01=0, n10=0, p=1.0)
    from scipy.stats import binomtest
    return dict(n01=n01, n10=n10, p=float(binomtest(min(n01, n10), n, 0.5).pvalue))


def aggregate_by_group(proba: np.ndarray, y: np.ndarray,
                       groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean posterior per group. One prediction per patient, not per slice."""
    g = np.asarray(groups)
    uniq = np.unique(g)
    P = np.zeros((len(uniq), proba.shape[1]), dtype=np.float64)
    Y = np.zeros(len(uniq), dtype=np.int64)
    for i, u in enumerate(uniq):
        m = g == u
        P[i] = proba[m].mean(axis=0)
        Y[i] = int(np.bincount(np.asarray(y)[m]).argmax())
    return P, Y, uniq


def confidence_weights(X_tr: np.ndarray, y_tr: np.ndarray, normal_idx: Optional[int],
                       gamma: float, fn_penalty: float,
                       seed: int = 42) -> Optional[np.ndarray]:
    """Eq. (5)-(6): (1 - p_true)^gamma, with tumour rows multiplied by fn_penalty.

    Note what this is and is not. The logistic probe that produces p_true is fitted
    on the same training rows it then scores, so p_true is in-sample and the weights
    reflect training-set difficulty. That is a stated design choice, not an
    accident, and it never sees validation or test rows.
    """
    if gamma == 0.0 and fn_penalty == 1.0:
        return None
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                             random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    p_true = clf.predict_proba(X_tr)[np.arange(len(y_tr)), np.asarray(y_tr)]
    w = (1.0 - p_true) ** gamma if gamma != 0.0 else np.ones(len(y_tr), dtype=np.float64)
    if fn_penalty != 1.0 and normal_idx is not None:
        w = w * np.where(np.asarray(y_tr) != normal_idx, fn_penalty, 1.0)
    w = w / max(float(np.mean(w)), 1e-8)
    return w.astype(np.float32)

# ---------------------------------------------------------------------------
# Learners -- thin adapters onto v4
# ---------------------------------------------------------------------------
# Each returns posteriors on whatever matrices it is handed. Crucially the
# signature takes X_eval, not X_test: during selection the caller passes the
# validation matrix twice, so a learner physically cannot see test rows before the
# ledger is frozen.

def fit_learner(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray,
                y_va: np.ndarray, X_eval: np.ndarray, n_classes: int, cfg: Cfg5,
                sample_weight: Optional[np.ndarray] = None,
                use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    base = B()
    t0 = time.time()
    if kind == "svm":
        pva, pev, _, _ = base.train_svm_proba(X_tr, y_tr, X_va, X_eval, cfg.svm_c,
                                              n_classes, sample_weight=sample_weight)
    elif kind == "xgb":
        pva, pev, _, _ = base.train_xgb_proba(X_tr, y_tr, X_va, y_va, X_eval,
                                              rounds=cfg.xgb_rounds,
                                              use_gpu=bool(use_gpu and cfg.xgb_gpu),
                                              n_classes=n_classes,
                                              sample_weight=sample_weight)
    elif kind == "lgb":
        pva, pev, name, _ = base.train_lgbm_proba(X_tr, y_tr, X_va, y_va, X_eval,
                                                  rounds=cfg.lgb_rounds,
                                                  try_gpu=cfg.lgbm_try_gpu,
                                                  n_classes=n_classes,
                                                  sample_weight=sample_weight)
        if name == "LightGBM_MISSING":
            raise RuntimeError("LightGBM is not installed in this interpreter")
    elif kind == "cat":
        pva, pev, name, _ = base.train_catboost_proba(X_tr, y_tr, X_va, y_va, X_eval,
                                                      rounds=cfg.cat_rounds,
                                                      n_classes=n_classes,
                                                      sample_weight=sample_weight)
        if name == "CatBoost_MISSING":
            raise RuntimeError("CatBoost is not installed in this interpreter")
    else:
        raise ValueError(f"unknown learner {kind!r}")
    return np.asarray(pva), np.asarray(pev), time.time() - t0

# ---------------------------------------------------------------------------
# Phase 1 -- selection. Validation only, by construction.
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Everything chosen on validation, plus the evidence for each choice."""
    bed: str
    spec: str
    learner: str
    thresholds: Optional[List[float]] = None
    ensemble_members: Optional[List[str]] = None
    ensemble_weights: Optional[List[float]] = None
    temperature: float = 1.0
    decision_rule: str = "argmax"
    valid_scores: Dict[str, float] = field(default_factory=dict)
    rule_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    grid: List[Dict[str, Any]] = field(default_factory=list)
    n_components: int = 0
    explained: float = 0.0
    config_digest: str = ""

    def to_json(self) -> Dict[str, Any]:
        return S._jsonable(asdict(self))


def select_on_validation(cfg: Cfg5, bed: Bed5, store: FeatureStore,
                         ledger: "S.SelectionLedger",
                         verbose: bool = True) -> SelectionResult:
    """Rank the full spec x learner grid on validation and fix every knob.

    The grid is scored with the *validation* cohort in both the "fit" and the
    "evaluate" position, so no test feature vector is loaded during this function.
    That is the mechanical difference from v4, whose loader returned train, valid
    and test together before the first model was fitted.

    What gets decided here, in order: which of the 7x5 configurations wins on
    validation balanced accuracy; the per-class thresholds; the ensemble weights;
    the calibration temperature; and which of the four decision rules to use. All
    five are written to the ledger with their evidence, and freeze() then pins them.
    """
    base = B()
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    M = bed.cost_matrix(cfg.c_miss, cfg.c_type, cfg.c_fp)

    for spec in cfg.specs:
        X_tr, y_tr = store.spec_features("train", spec, 1)
        X_va, y_va = store.spec_features("valid", spec, cfg.tta)
        rep, Z_tr = fit_representation(X_tr, cfg.standardize, cfg.pca_var,
                                       dims=spec_dims(spec), seed=cfg.random_state)
        Z_va = rep.transform(X_va)

        sw = None
        if cfg.use_hard_weighting:
            sw = confidence_weights(Z_tr, y_tr, bed.normal_idx, cfg.hard_weight_gamma,
                                    cfg.hard_weight_fn_penalty, seed=cfg.random_state)

        probs_va: Dict[str, np.ndarray] = {}
        for L in cfg.learners:
            try:
                # X_eval is the validation matrix: selection never touches test
                pva, _pev, secs = fit_learner(L, Z_tr, y_tr, Z_va, y_va, Z_va,
                                              bed.n_classes, cfg, sample_weight=sw)
            except Exception as e:                      # a missing learner is not fatal
                if verbose:
                    print(f"  [SKIP] {spec_label(spec):18s} {L:4s}: {e}")
                continue
            probs_va[L] = pva
            row = _grid_row(bed, cfg, spec, L, pva, y_va, M, rep, secs)
            rows.append(row)
            if best is None or row["valid_bal_acc"] > best["valid_bal_acc"]:
                best = row
            if verbose:
                print(f"  {spec_label(spec):18s} {L:4s}  valid acc={row['valid_acc']*100:6.2f} "
                      f"bal={row['valid_bal_acc']*100:6.2f} k={rep.n_components:4d} "
                      f"({secs:5.1f}s)")

        if cfg.optimize_ensemble and len(probs_va) > 1:
            names = sorted(probs_va)
            w, thr, sc = base.optimize_ensemble_weights(
                [probs_va[n] for n in names], names, y_va, bed.n_classes,
                bed.normal_idx, cfg.threshold_metric, cfg.min_tumor_recall)
            pens = base.weighted_average_probs([probs_va[n] for n in names], list(w))
            row = _grid_row(bed, cfg, spec, "ens", pens, y_va, M, rep, 0.0,
                            thr_override=thr)
            row["ensemble_members"] = names
            row["ensemble_weights"] = [float(x) for x in w]
            row["ensemble_valid_score"] = float(sc)
            row["thresholds"] = [float(x) for x in thr]
            rows.append(row)
            if best is None or row["valid_bal_acc"] > best["valid_bal_acc"]:
                best = row
            if verbose:
                print(f"  {spec_label(spec):18s} ens   valid acc={row['valid_acc']*100:6.2f} "
                      f"bal={row['valid_bal_acc']*100:6.2f}  w={dict(zip(names, np.round(w, 2)))}")

    if best is None:
        raise RuntimeError("no configuration completed on validation; nothing to select")

    # -- which decision rule? decided on validation, on the winning config only ---
    rule_scores = {k: v for k, v in best["rule_scores"].items()}
    if cfg.decision_rule == "auto":
        # tie-break on expected risk, because two rules can share balanced accuracy
        # while differing on the error the cost matrix says matters
        chosen = max(rule_scores, key=lambda r: (round(rule_scores[r]["bal_acc"], 6),
                                                 -rule_scores[r]["exp_risk"]))
    else:
        chosen = cfg.decision_rule
        if chosen not in rule_scores:
            raise ValueError(f"decision rule {chosen!r} was not scored on validation")

    res = SelectionResult(
        bed=bed.key, spec=best["spec"], learner=best["learner"],
        thresholds=best.get("thresholds"),
        ensemble_members=best.get("ensemble_members"),
        ensemble_weights=best.get("ensemble_weights"),
        temperature=float(best["temperature"]), decision_rule=chosen,
        valid_scores={k: v for k, v in best.items() if k.startswith("valid_")},
        rule_scores=rule_scores, grid=rows,
        n_components=int(best["n_components"]), explained=float(best["explained"]))

    ev = (f"validation balanced accuracy {best['valid_bal_acc']*100:.2f} over "
          f"{len(rows)} configurations")
    ledger.record("feature_spec", res.spec, evidence=ev)
    ledger.record("learner", res.learner, evidence=ev)
    ledger.record("retained_variance", cfg.pca_var,
                  evidence=f"{res.n_components} components, {res.explained:.4f} variance")
    ledger.record("standardization", cfg.standardize, evidence="protocol default")
    # The ledger records the setting the selection was actually made at, not the one
    # requested. On both four-class beds validation rows come out of the never-augmented
    # Train cache, so cfg.tta=3 is served un-augmented; sealing "tta=3" there would pin a
    # condition the selection never ran under, and stage_report would then reopen a
    # ledger that misdescribes its own evidence.
    tta_va = store.effective_tta("valid", requested=int(cfg.tta))
    eff_va = store.effective_tta_scalar("valid", requested=int(cfg.tta))
    # The views go in the EVIDENCE, not in the value. freeze() hashes only ("knob",
    # "value"), so the digest stays a function of settings alone: `views` is derived from
    # PRE.TTA_VIEWS, which is a description of code_final_fixed_v4.tta_variants rather
    # than a knob, and a digest that moved when a description was corrected would be
    # pinning the wrong thing. It cannot drift into a lie either --
    # PRE.assert_tta_views_match_reference() checks it against v4's source and raises.
    # What belongs in a sealed ledger is that the averaging was over pose and not over
    # intensity, because that is the condition a reviewer would want held constant.
    ledger.record("tta", dict(requested=int(cfg.tta), effective_valid=eff_va,
                              effective_train=1, substituted=tta_va["substituted"]),
                  evidence=("protocol default tta=%d; train is always tta=1; validation "
                            "served at %s%s; views=%s (identity included, every other "
                            "view geometric -- no intensity-based view, and rot_p7 "
                            "blanks %.2f%% of the frame with NEAREST/black fill before "
                            "the pinned resize)"
                            % (cfg.tta, eff_va,
                               "" if not tta_va["substituted"]
                               else " -- " + "; ".join(tta_va["notes"]),
                               ",".join(PRE.tta_view_names(int(cfg.tta))),
                               100.0 * PRE.TTA_WEDGE_AUDIT["frame_fraction_blanked"])))
    if res.thresholds is not None:
        ledger.record("thresholds", res.thresholds,
                      evidence=f"tuned on validation for {cfg.threshold_metric}")
    if res.ensemble_weights is not None:
        ledger.record("ensemble_weights", res.ensemble_weights,
                      evidence="simplex grid search on validation")
    ledger.record("temperature", res.temperature, evidence="validation NLL minimum")
    ledger.record("cost_matrix", dict(c_miss=cfg.c_miss, c_type=cfg.c_type, c_fp=cfg.c_fp),
                  evidence="fixed a priori from clinical asymmetry, not tuned")
    ledger.record("decision_rule", chosen,
                  evidence="; ".join(f"{r}: bal_acc={v['bal_acc']*100:.2f}, "
                                     f"risk={v['exp_risk']:.4f}"
                                     for r, v in sorted(rule_scores.items())))
    ledger.record("sample_weighting",
                  dict(enabled=cfg.use_hard_weighting, gamma=cfg.hard_weight_gamma,
                       fn_penalty=cfg.hard_weight_fn_penalty),
                  evidence="protocol default from the manuscript")
    # The seal note also states which corpus this cohort came from and how much of it
    # the OTHER bed trains on. It goes in the note and not in a `record` call because
    # freeze() hashes only (knob, value): a measured fact recorded as a knob would put
    # 3038/3064 into the config digest, and correcting a transcription error there would
    # invalidate every cached matrix pinned to that digest. The note is unhashed, which
    # is exactly the right place for a description of the cohort rather than a choice
    # about it.
    _ov = S.cohort_overlap(bed.key)
    res.config_digest = ledger.freeze(
        note=f"selection closed on bed={bed.key}; {len(rows)} configurations scored on "
             f"validation only; test cohort still unread"
             + (f". Cohort context: shares pixels with {_ov['corpus_shared_with']} -- "
                f"{_ov['cohort_test_in_other_dev']} of this bed's test files are in that "
                f"corpus's development pool, so independence of this cohort is a claim "
                f"about THIS bed and not about a table that also quotes the other one"
                if _ov["corpus_shared_with"] else ""))
    return res


def _grid_row(bed: Bed5, cfg: Cfg5, spec: str, learner: str, pva: np.ndarray,
              y_va: np.ndarray, M: np.ndarray, rep: Representation,
              secs: float, thr_override: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Score one configuration on validation under all four decision rules."""
    base = B()
    if thr_override is None:
        thr, thr_score = base.tune_thresholds(y_va, pva, cfg.threshold_metric,
                                              cfg.min_tumor_recall, bed.normal_idx,
                                              bed.n_classes)
    else:
        thr = np.asarray(thr_override, dtype=np.float64)
        thr_score = float(base.score_predictions(
            y_va, base.predict_with_thresholds(pva, thr, bed.normal_idx),
            cfg.threshold_metric if cfg.threshold_metric != "f1_pos" else "bal_acc"))
    T = temperature_fit(pva, y_va) if cfg.calibrate else 1.0
    pva_c = temperature_apply(pva, T)

    rule_scores: Dict[str, Dict[str, float]] = {}
    for rule in ("argmax", "threshold", "bayes", "bayes_cal"):
        pred = apply_rule(pva, rule, thresholds=thr, normal_idx=bed.normal_idx,
                          cost=M, temperature=T)
        sb = score_block(bed, y_va, pred)
        rule_scores[rule] = dict(acc=sb["acc"], bal_acc=sb["bal_acc"], f1=sb["f1"],
                                 fnr_macro=sb["fnr_macro"],
                                 fnr_pres=sb.get("fnr_pres", float("nan")),
                                 exp_risk=expected_risk(y_va, pred, M))

    argmax_sb = score_block(bed, y_va, pva.argmax(axis=1), prefix="valid_")
    row: Dict[str, Any] = dict(
        bed=bed.key, spec=spec, spec_label=spec_label(spec), learner=learner,
        learner_label=LEARNER_LABEL.get(learner, learner),
        n_components=rep.n_components, explained=rep.explained,
        fit_seconds=float(secs), temperature=float(T),
        thresholds=[float(x) for x in thr], threshold_score=float(thr_score),
        valid_ece_raw=ece(pva, y_va), valid_ece_cal=ece(pva_c, y_va),
        rule_scores=rule_scores)
    row.update(argmax_sb)
    return row

# ---------------------------------------------------------------------------
# Phase 2 -- reporting. Gated on the ledger.
# ---------------------------------------------------------------------------
# The two phases are separate process invocations by design. `--stage select`
# writes selection.json and seals the ledger; `--stage report` reopens both and
# has no code path that can change a knob. The first thing it does with the test
# manifest is ask the ledger for permission, and the ledger raises unless the seal
# is already in place.

def _selection_path(cfg: Cfg5) -> str:
    return os.path.join(cfg.run_dir, "selection.json")


def save_selection(cfg: Cfg5, res: SelectionResult) -> str:
    p = _selection_path(cfg)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(selection=res.to_json(), config=cfg.to_json(),
                       config_digest=cfg.digest()), f, ensure_ascii=False, indent=2)
    return p


def load_selection(cfg: Cfg5) -> SelectionResult:
    p = _selection_path(cfg)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} is missing. The reporting stage does not choose anything; it only "
            f"executes what `--stage select` already sealed. Run selection first.")
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    s = d["selection"]
    known = {f for f in SelectionResult.__dataclass_fields__}
    return SelectionResult(**{k: v for k, v in s.items() if k in known})


def _fit_final(cfg: Cfg5, bed: Bed5, store: FeatureStore, res: SelectionResult,
               X_eval: np.ndarray,
               verbose: bool = True) -> Tuple[np.ndarray, Representation, Dict[str, Any]]:
    """Refit the sealed configuration and score one evaluation matrix.

    `X_eval` is raw concatenated embeddings for the cohort being scored. Nothing in
    here consults it except the final predict, and no threshold, weight or
    temperature is recomputed: all three come from `res`, which came from validation.
    """
    spec = res.spec
    X_tr, y_tr = store.spec_features("train", spec, 1)
    X_va, y_va = store.spec_features("valid", spec, cfg.tta)

    if cfg.refit_on_train_valid:
        # Legitimate and common -- more training data for the final model -- but it
        # changes what the reported number means, so it is off by default and always
        # written into the artifact.
        X_fit = np.concatenate([X_tr, X_va], axis=0)
        y_fit = np.concatenate([y_tr, y_va], axis=0)
    else:
        X_fit, y_fit = X_tr, y_tr

    rep, Z_fit = fit_representation(X_fit, cfg.standardize, cfg.pca_var,
                                    dims=spec_dims(spec), seed=cfg.random_state)
    Z_va = rep.transform(X_va)
    Z_ev = rep.transform(X_eval)
    sw = None
    if cfg.use_hard_weighting:
        sw = confidence_weights(Z_fit, y_fit, bed.normal_idx, cfg.hard_weight_gamma,
                                cfg.hard_weight_fn_penalty, seed=cfg.random_state)

    meta: Dict[str, Any] = dict(spec=spec, learner=res.learner,
                                n_components=rep.n_components, explained=rep.explained,
                                refit_on_train_valid=bool(cfg.refit_on_train_valid),
                                n_fit_rows=int(len(y_fit)))
    if res.learner == "ens":
        members = list(res.ensemble_members or [])
        weights = list(res.ensemble_weights or [])
        if len(members) != len(weights):
            raise RuntimeError("sealed ensemble has a different number of members and weights")
        parts, times = [], {}
        for L in members:
            _pva, pev, secs = fit_learner(L, Z_fit, y_fit, Z_va, y_va, Z_ev,
                                          bed.n_classes, cfg, sample_weight=sw)
            parts.append(pev)
            times[L] = float(secs)
            if verbose:
                print(f"    refit {L:4s} ({secs:5.1f}s)")
        proba = B().weighted_average_probs(parts, weights)
        meta.update(ensemble_members=members, ensemble_weights=weights,
                    fit_seconds=times)
    else:
        _pva, proba, secs = fit_learner(res.learner, Z_fit, y_fit, Z_va, y_va, Z_ev,
                                        bed.n_classes, cfg, sample_weight=sw)
        meta["fit_seconds"] = {res.learner: float(secs)}
        if verbose:
            print(f"    refit {res.learner:4s} ({secs:5.1f}s)")
    return np.asarray(proba), rep, meta

def report_on_test(cfg: Cfg5, bed: Bed5, store: FeatureStore,
                   ledger: "S.SelectionLedger", res: SelectionResult,
                   purpose: str = "primary_result",
                   verbose: bool = True) -> Dict[str, Any]:
    """Read the sealed cohort once, apply the sealed configuration, report.

    Note the order of the first two statements. The authorization call comes *before*
    the features are touched, so if the ledger is not frozen the test .npy is never
    even opened. That is the difference between a protocol and a promise.
    """
    ledger.authorize_test_read(bed.key, purpose)

    m_te = bed.split("test")
    X_te, y_te = store.spec_features("test", res.spec, cfg.tta)
    # Read back what the test matrix was actually built at. On the four-class beds the
    # Test folder holds both settings, so this should equal cfg.tta; if it does not, the
    # reported row is not the condition the ledger sealed and the caller must be able to
    # see that in the artifact rather than infer it.
    tta_te = store.effective_tta("test", spec=res.spec, requested=int(cfg.tta))
    proba, rep, meta = _fit_final(cfg, bed, store, res, X_te, verbose=verbose)

    M = bed.cost_matrix(cfg.c_miss, cfg.c_type, cfg.c_fp)
    pred = apply_rule(proba, res.decision_rule,
                      thresholds=np.asarray(res.thresholds) if res.thresholds else None,
                      normal_idx=bed.normal_idx, cost=M, temperature=res.temperature)

    # `meta` already carries `spec` and `learner`: _fit_final records what it actually
    # fitted, not what was asked for, so those two keys arrive through the splat on the
    # last line and must not be named again here. Naming them twice is what raised
    # `TypeError: dict() got multiple values for keyword argument 'spec'`, and it raised
    # it *after* authorize_test_read had already spent this cohort's one read.
    out: Dict[str, Any] = dict(
        bed=bed.key, purpose=purpose, spec_label=spec_label(res.spec),
        learner_label=LEARNER_LABEL.get(res.learner, res.learner),
        decision_rule=res.decision_rule, temperature=float(res.temperature),
        n_test=int(m_te.n), n_test_groups=int(m_te.n_groups()),
        group_unit=bed.group_unit, manifest_digest=m_te.digest(),
        tta_requested=int(cfg.tta),
        tta_effective=(tta_te["effective"][0] if tta_te["uniform"] and tta_te["effective"]
                       else tta_te["effective"]),
        tta_substituted=bool(tta_te["substituted"]),
        tta_note="; ".join(tta_te["notes"]),
        config_digest=ledger.frozen_digest, **meta)
    out.update(score_block(bed, y_te, pred, prefix="test_"))
    out["test_exp_risk"] = expected_risk(y_te, pred, M)
    out["test_ece_raw"] = ece(proba, y_te)
    out["test_ece_cal"] = ece(temperature_apply(proba, res.temperature), y_te)

    # -- every rule on the same posteriors, so the selected one is auditable -----
    # These are *reported*, not chosen from: the choice was made on validation and is
    # already in the ledger. Printing the others lets a reader see whether the
    # validation decision transferred, which is exactly the check a reviewer wants
    # and the thing a single headline number hides.
    rules: Dict[str, Any] = {}
    for r in ("argmax", "threshold", "bayes", "bayes_cal"):
        pr = apply_rule(proba, r,
                        thresholds=np.asarray(res.thresholds) if res.thresholds else None,
                        normal_idx=bed.normal_idx, cost=M, temperature=res.temperature)
        sb = score_block(bed, y_te, pr)
        rules[r] = dict(acc=sb["acc"], bal_acc=sb["bal_acc"], f1=sb["f1"],
                        fnr_macro=sb["fnr_macro"],
                        fnr_pres=sb.get("fnr_pres", float("nan")),
                        exp_risk=expected_risk(y_te, pr, M),
                        selected_on_valid=(r == res.decision_rule),
                        valid_bal_acc=res.rule_scores.get(r, {}).get("bal_acc"))
    out["rules_on_test"] = rules

    # -- intervals -------------------------------------------------------------
    groups = bed.groups("test")
    n_units = int(len(np.unique(groups)))
    if cfg.cluster_bootstrap and n_units < m_te.n:
        out["ci_cluster"] = cluster_bootstrap(y_te, pred, groups, cfg.bootstrap_n,
                                              cfg.random_state)
    if cfg.report_iid_bootstrap_too or n_units >= m_te.n:
        out["ci_iid"] = iid_bootstrap(y_te, pred, cfg.bootstrap_n, cfg.random_state)
    if "ci_cluster" in out and "ci_iid" in out:
        cw = out["ci_cluster"]["hi"] - out["ci_cluster"]["lo"]
        iw = out["ci_iid"]["hi"] - out["ci_iid"]["lo"]
        out["ci_width_ratio"] = float(cw / iw) if iw > 0 else float("nan")
        out["ci_note"] = (
            f"{m_te.n} slices from {n_units} {bed.group_unit}s: the patient-clustered "
            f"interval is {out['ci_width_ratio']:.2f}x the width of the i.i.d. one. The "
            f"i.i.d. interval is reported only to show the size of the understatement.")

    # -- one prediction per patient -------------------------------------------
    if cfg.patient_level and n_units < m_te.n:
        Pg, Yg, ug = aggregate_by_group(proba, y_te, groups)
        pg = apply_rule(Pg, res.decision_rule,
                        thresholds=np.asarray(res.thresholds) if res.thresholds else None,
                        normal_idx=bed.normal_idx, cost=M, temperature=res.temperature)
        out.update(score_block(bed, Yg, pg, prefix="patient_"))
        out["n_patients"] = int(len(ug))
        out["patient_exp_risk"] = expected_risk(Yg, pg, M)

    out["slice_predictions"] = dict(files=m_te.files, y_true=[int(v) for v in y_te],
                                    y_pred=[int(v) for v in pred],
                                    groups=[str(g) for g in groups])
    if verbose:
        print(f"  test  acc={out['test_acc']*100:6.2f}  bal={out['test_bal_acc']*100:6.2f}  "
              f"risk={out['test_exp_risk']:.4f}  rule={res.decision_rule}")
        if "ci_cluster" in out:
            c = out["ci_cluster"]
            print(f"        patient-clustered 95% CI [{c['lo']*100:.2f}, {c['hi']*100:.2f}] "
                  f"over {c['n_units']} {bed.group_unit}s")
    return out

# ---------------------------------------------------------------------------
# Extra backbones, for the competitors only
# ---------------------------------------------------------------------------
# v4's build_extractor covers the three backbones of the proposed method. The
# published methods v5 reproduces need four more (GoogLeNet, VGG-19, DenseNet-169,
# ResNeXt-50). They are built here rather than by editing v4, because v4 is the
# reference implementation behind the submitted numbers and should stay untouched.
# Same weights source, same preprocessing, same global-average-pool convention.

EXTRA_DIM = {"googlenet": 1024, "vgg19": 4096, "densenet169": 1664, "resnext50": 2048,
             "resnet50": 2048, "efficientnet_b0": 1280}
ALL_DIM = dict(BACKBONE_DIM)
ALL_DIM.update(EXTRA_DIM)


def build_extractor_v5(backbone: str, device: Any, img_size: int = 224):
    """Frozen ImageNet feature extractor. Delegates to v4 where v4 knows the model."""
    import torch.nn as nn
    from torchvision import models

    b = backbone.lower().strip()
    if b in ("resnet18", "densenet121", "vit_b_16", "vit", "vitb16", "resnet50",
             "efficientnet_b0", "efficientnet", "effb0", "densenet", "d121",
             # The xrv names stay reachable here so v5 does not silently lose a model
             # v4 can build, but they are NOT pinned in preprocess_v5: torchxrayvision
             # rescales internally (gray*2048-1024, one channel), so ImageNet mean/std
             # in front of it would scale twice. No spec and no baseline uses them, so
             # the first call touching one raises from PRE.norm_for with that reason.
             # Deliberate, not an omission -- see preprocess_v5._UNSUPPORTED.
             "xraytorchvision", "xrv", "xrv_densenet"):
        return B().build_extractor(b, device, img_size)

    if b in ("googlenet", "inception_v1"):
        # Deepak & Ameer 2019 and Sekhar et al. 2021 both take GoogLeNet's
        # 1024-d pool5 output, which is what dropping the fc layer leaves.
        net = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT,
                               aux_logits=True, init_weights=False)
        net.aux1 = None
        net.aux2 = None
        net.fc = nn.Identity()
        ex, dim = net, 1024
    elif b in ("vgg19", "vgg_19"):
        # Swati et al. 2019 read the first fully connected layer (4096-d). This is
        # the frozen-feature variant of that method, not their block-wise
        # fine-tuning; the difference is stated in the baseline's `notes`.
        net = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        net.classifier = nn.Sequential(*list(net.classifier.children())[:2])
        ex, dim = net, 4096
    elif b in ("densenet169", "d169"):
        net = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        ex = nn.Sequential(net.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        dim = 1664
    elif b in ("resnext50", "resnext50_32x4d"):
        net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        ex = nn.Sequential(*list(net.children())[:-1])
        dim = 2048
    else:
        raise ValueError(f"unknown backbone {backbone!r}")

    import torch
    ex = ex.to(device).eval()
    if getattr(device, "type", "cpu") == "cuda":
        ex = ex.to(memory_format=torch.channels_last)
    return ex, dim

# ---------------------------------------------------------------------------
# Competitors -- rejection ground (1)
# ---------------------------------------------------------------------------
# The rejection said the manuscript "does not yet establish how the proposed method
# compares to the most relevant published work under directly comparable conditions:
# the same cohorts, training and validation splits, preprocessing, evaluation
# criteria, and held-out test cases." Quoting each paper's own accuracy does not
# establish that, because each was measured on a different split of a different
# cohort with different preprocessing.
#
# So each entry below is a *reimplementation of the competitor's classifier stage*
# run through this file's manifest, this file's preprocessing, this file's decision
# rule and this file's sealed test cohort. Every row of the baseline table therefore
# differs from the proposed row in exactly one respect: the method.
#
# Two honesty constraints, both recorded in `notes` and reprinted in the artifact:
#
#   * Where the original fine-tuned its backbone, the frozen-feature version here is
#     not that method. It is that method's *architecture and classifier* under this
#     paper's frozen-embedding protocol -- which is the comparison the proposed
#     method's claim actually needs, since the claim is about frozen embeddings. The
#     published accuracy is carried in `reported` for reference and must never be
#     put in the same column as a measured number.
#   * Any hyperparameter the original left unspecified is set to the library default
#     and named in `notes`, so nobody has to guess whether it was tuned here. None of
#     them were tuned on the test cohort -- they cannot be, the ledger forbids it.

@dataclass
class Baseline:
    key: str
    label: str
    citation: str
    spec: str                       # one or more backbones, '+'-joined
    head: str                       # svm | xgb | lgb | cat | rf | knn | linsvm | logreg
    standardize: str = "per_dim"
    pca_var: float = 0.0            # 0 means no PCA, matching most published recipes
    head_kwargs: Dict[str, Any] = field(default_factory=dict)
    tta: int = 1
    reported: Optional[float] = None
    notes: str = ""

    def title(self) -> str:
        return f"{self.label} [{self.citation}]"


BASELINES: Dict[str, Baseline] = {
    "deepak2019": Baseline(
        key="deepak2019", label="GoogLeNet + SVM", citation="Deepak & Ameer 2019",
        spec="googlenet", head="svm", pca_var=0.0,
        head_kwargs=dict(C=1.0, gamma="scale"), reported=0.9810,
        notes="Transfer-learned GoogLeNet features into an SVM. The original used the "
              "figshare 3-class cohort with 5-fold patient-level CV; here the same "
              "feature/classifier pair runs on this paper's splits. RBF kernel with "
              "library defaults (C=1, gamma='scale') since the paper does not state them."),
    "sekhar2021": Baseline(
        key="sekhar2021", label="GoogLeNet + k-NN", citation="Sekhar et al. 2021",
        spec="googlenet", head="knn", pca_var=0.0, head_kwargs=dict(n_neighbors=5),
        reported=0.9880,
        notes="Same GoogLeNet embedding, k-NN head. k=5, Euclidean, uniform weights."),
    "swati2019": Baseline(
        key="swati2019", label="VGG-19 fc1 + SVM", citation="Swati et al. 2019",
        spec="vgg19", head="svm", pca_var=0.0, head_kwargs=dict(C=1.0, gamma="scale"),
        reported=0.9482,
        notes="The original block-wise fine-tuned VGG-19. This is the frozen-feature "
              "variant of the same architecture and head, which is the comparison "
              "relevant to a frozen-embedding claim; it is NOT a reproduction of their "
              "fine-tuning and the reported 94.82% is not comparable to this row."),
    "kang2021": Baseline(
        key="kang2021", label="Concat(3 CNN) + SVM-RBF", citation="Kang et al. 2021",
        spec="densenet169+resnext50+resnet18", head="svm", pca_var=0.0,
        head_kwargs=dict(C=1.0, gamma="scale"), reported=None,
        notes="Kang et al. concatenate several deep feature sets and select the best "
              "classifier from nine; their top performer on multi-class data was "
              "SVM-RBF. Backbones are the three of their pool with ImageNet weights in "
              "torchvision, unnormalized concatenation, no PCA -- exactly their recipe. "
              "This is the closest published analogue of the proposed method and the "
              "row a reader should compare against first."),
    "kang2021_pca": Baseline(
        key="kang2021_pca", label="Concat(3 CNN) + PCA + SVM-RBF",
        citation="Kang et al. 2021, + this paper's PCA stage",
        spec="densenet169+resnext50+resnet18", head="svm", pca_var=0.95,
        head_kwargs=dict(C=1.0, gamma="scale"),
        notes="Kang's fusion with this paper's standardize-then-PCA stage bolted on. "
              "The gap between this row and the previous one isolates what the "
              "dimensionality-reduction stage contributes, independent of which "
              "backbones are fused."),
    "vit_rf": Baseline(
        key="vit_rf", label="ViT-B/16 + PCA + Random Forest",
        citation="frozen-transformer baseline", spec="vit_b_16", head="rf",
        pca_var=0.95, head_kwargs=dict(n_estimators=500, min_samples_leaf=1),
        notes="A frozen-transformer control: same embedding as the proposed method's "
              "ViT branch, a head from outside the gradient-boosting family. Shows "
              "whether the gain is in the representation or in the learner."),
    "linear_probe": Baseline(
        key="linear_probe", label="Linear probe (logistic on R18+D121+ViT)",
        citation="standard frozen-feature control",
        spec="resnet18+densenet121+vit_b_16", head="logreg", pca_var=0.0,
        head_kwargs=dict(C=1.0, max_iter=3000),
        notes="The control every frozen-feature paper owes its reader: multinomial "
              "logistic regression on the same 2,304-d concatenation the proposed "
              "method uses. If the elaborate pipeline does not beat this, the pipeline "
              "is not what is doing the work."),
}


def competitor_availability(cfg: Cfg5, bed: Bed5, store: FeatureStore,
                            keys: Optional[Sequence[str]] = None
                            ) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Which competitors can run right now, without reading or extracting anything.

    Returns (backbones_with_a_cache, {backbone: why_not}, runnable_baseline_keys).

    Resolution goes through a throwaway CacheIndex rather than through
    FeatureStore.backbone_features, for three reasons. A probe must not trigger the
    extraction that --extract-missing authorizes. It must not write entries into
    store.binds, which is the run's record of what the *experiment* bound to. And it
    must not read the sealed cohort: resolve() globs filenames and reads .npy headers
    for their shape, so not one test row is loaded and no ledger authorization is
    owed. What it checks is exactly what _from_existing checks -- that a cache exists
    covering each source folder in full -- minus the gather.
    """
    ks = list(keys) if keys else list(BASELINES)
    probe = S.CacheIndex(store.index.cache_dirs, img_size=cfg.img_size)
    bbs = sorted({b for k in ks for b in BASELINES[k].spec.split("+")})
    have: List[str] = []
    need: Dict[str, str] = {}
    for bb in bbs:
        try:
            for role in S.ROLES:
                for sp in sorted(set(bed.split(role).source_splits)):
                    probe.resolve(bb, S._cache_split_key(sp),
                                  int(store.source_sizes[sp]), tag=f"probe/{bb}/{sp}")
            have.append(bb)
        except Exception as e:
            need[bb] = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    runnable = [k for k in ks
                if all(b in have for b in BASELINES[k].spec.split("+"))]
    return have, need, runnable


def fit_baseline_head(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray,
                      n_classes: int, seed: int, kwargs: Dict[str, Any],
                      cfg: Cfg5, X_va: Optional[np.ndarray] = None,
                      y_va: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Fit a competitor's classifier stage and return posteriors on X_ev.

    A competitor's own hyperparameters win here. In particular `svm` builds an
    sklearn SVC with the C and gamma named in the Baseline rather than routing through
    v4's train_svm_proba, which would silently impose the proposed method's C=1.5 and
    turn the comparison into a comparison of two settings of one method.
    """
    t0 = time.time()
    if kind == "svm":
        from sklearn.svm import SVC
        kw = dict(C=1.0, gamma="scale")
        kw.update(kwargs or {})
        clf = SVC(kernel="rbf", probability=True, random_state=seed, **kw)
    elif kind in ("xgb", "lgb", "cat"):
        Xv = X_va if X_va is not None else X_ev
        yv = y_va if y_va is not None else y_tr[:0]
        _pva, pev, secs = fit_learner(kind, X_tr, y_tr, Xv, yv, X_ev, n_classes, cfg,
                                      sample_weight=None)
        return np.asarray(pev), float(secs)
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **kwargs)
    elif kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_jobs=-1, **kwargs)
    elif kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=seed, n_jobs=-1, **kwargs)
    elif kind == "linsvm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC
        clf = CalibratedClassifierCV(LinearSVC(random_state=seed, **kwargs), cv=3)
    else:
        raise ValueError(f"unknown baseline head {kind!r}")
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_ev)
    return B().ensure_proba_matrix(np.asarray(p), n_classes), time.time() - t0

def run_baselines(cfg: Cfg5, bed: Bed5, store: FeatureStore,
                  ledger: "S.SelectionLedger", res: SelectionResult,
                  keys: Optional[Sequence[str]] = None,
                  verbose: bool = True) -> List[Dict[str, Any]]:
    """Every competitor on the same manifest, the same cohort, the same rule.

    Deliberately *not* given the freedom the proposed method got. A competitor gets
    no per-class threshold search, no ensemble weight optimization and no temperature
    fit, because the proposed method's copies of those were selected on validation and
    are already sealed. What a competitor gets is: the same training rows, the same
    validation rows if its learner needs early stopping, the same preprocessing, and
    then argmax -- plus, for completeness, its numbers under the sealed decision rule
    so a reader can check the rule is not what is carrying the result.

    Each baseline is a separate ledger purpose, so the read counter shows one read per
    competitor rather than pretending they were all one look.
    """
    keys = list(keys) if keys else list(BASELINES)
    rows: List[Dict[str, Any]] = []
    M = bed.cost_matrix(cfg.c_miss, cfg.c_type, cfg.c_fp)
    groups = bed.groups("test")
    y_ref: Optional[np.ndarray] = None
    proba_by_key: Dict[str, np.ndarray] = {}
    pred_by_key: Dict[str, np.ndarray] = {}

    for k in keys:
        bl = BASELINES[k]
        if verbose:
            print(f"  [{bl.key}] {bl.title()}")
        # Features first, authorization second. A competitor whose backbone has no
        # cache never touches the cohort, so it must not consume a test read: the
        # ledger's counter is the evidence for "read once per comparison", and a read
        # logged for a comparison that did not happen makes it evidence of nothing.
        # It would also be self-defeating -- reads persist, so burning one here would
        # make the re-run after extraction raise TestCohortViolation.
        try:
            X_tr, y_tr = store.spec_features("train", bl.spec, 1)
            X_va, y_va = store.spec_features("valid", bl.spec, bl.tta)
        except Exception as e:
            rows.append(dict(bed=bed.key, kind="baseline", key=bl.key, label=bl.label,
                             citation=bl.citation, spec=bl.spec, head=bl.head,
                             status="features_unavailable", error=str(e),
                             test_cohort_read=False, notes=bl.notes))
            if verbose:
                print(f"      SKIP (no test read consumed): {e}")
            continue

        ledger.authorize_test_read(bed.key, f"baseline:{bl.key}")
        X_te, y_te = store.spec_features("test", bl.spec, bl.tta)

        rep, Z_tr = fit_representation(X_tr, bl.standardize, bl.pca_var,
                                       dims=spec_dims(bl.spec), seed=cfg.random_state)
        Z_va, Z_te = rep.transform(X_va), rep.transform(X_te)
        proba, secs = fit_baseline_head(bl.head, Z_tr, y_tr, Z_te, bed.n_classes,
                                        cfg.random_state, dict(bl.head_kwargs), cfg,
                                        X_va=Z_va, y_va=y_va)
        pred = proba.argmax(axis=1)
        y_ref = y_te if y_ref is None else y_ref
        proba_by_key[bl.key] = proba
        pred_by_key[bl.key] = pred

        row: Dict[str, Any] = dict(
            bed=bed.key, kind="baseline", key=bl.key, label=bl.label,
            citation=bl.citation, spec=bl.spec, spec_label=bl.spec,
            head=bl.head, standardize=bl.standardize, pca_var=bl.pca_var,
            n_components=rep.n_components, explained=rep.explained,
            head_kwargs=dict(bl.head_kwargs),
            tta_requested=int(bl.tta),
            # A competitor row is only a same-condition row if the condition it claims is
            # the condition it ran under. baseline_tta=1 against a proposed row at tta=3
            # is the single-respect difference the whole table rests on, so the effective
            # setting is read back out of the bind log per competitor rather than copied
            # from the request.
            tta_effective_test=store.effective_tta_scalar("test", spec=bl.spec,
                                                          requested=int(bl.tta)),
            tta_effective_valid=store.effective_tta_scalar("valid", spec=bl.spec,
                                                           requested=int(bl.tta)),
            tta_effective_train=store.effective_tta_scalar("train", spec=bl.spec,
                                                           requested=1),
            tta_substituted=bool(store.effective_tta("test", spec=bl.spec,
                                                     requested=int(bl.tta))["substituted"]
                                 or store.effective_tta("valid", spec=bl.spec,
                                                        requested=int(bl.tta))["substituted"]),
            decision_rule="argmax", fit_seconds=float(secs), status="ok",
            test_cohort_read=True,
            n_test=int(bed.split("test").n), manifest_digest=bed.split("test").digest(),
            reported_in_original=bl.reported,
            reported_is_not_comparable=True, notes=bl.notes)
        row.update(score_block(bed, y_te, pred, prefix="test_"))
        row["test_exp_risk"] = expected_risk(y_te, pred, M)
        row["test_ece_raw"] = ece(proba, y_te)

        # the same competitor under the proposed method's sealed rule
        pred_rule = apply_rule(proba, res.decision_rule,
                               thresholds=np.asarray(res.thresholds) if res.thresholds else None,
                               normal_idx=bed.normal_idx, cost=M,
                               temperature=res.temperature)
        sb = score_block(bed, y_te, pred_rule)
        row["under_sealed_rule"] = dict(rule=res.decision_rule, acc=sb["acc"],
                                        bal_acc=sb["bal_acc"], f1=sb["f1"],
                                        exp_risk=expected_risk(y_te, pred_rule, M))

        n_units = int(len(np.unique(groups)))
        if cfg.cluster_bootstrap and n_units < len(y_te):
            row["ci_cluster"] = cluster_bootstrap(y_te, pred, groups, cfg.bootstrap_n,
                                                  cfg.random_state)
        else:
            row["ci_iid"] = iid_bootstrap(y_te, pred, cfg.bootstrap_n, cfg.random_state)
        # kept so compare_against_baselines() can run a *paired* test rather than
        # comparing two summary statistics
        row["slice_predictions"] = dict(y_true=[int(v) for v in y_te],
                                       y_pred=[int(v) for v in pred])
        rows.append(row)
        if verbose:
            print(f"      acc={row['test_acc']*100:6.2f} bal={row['test_bal_acc']*100:6.2f} "
                  f"k={rep.n_components:4d} ({secs:5.1f}s)")
    return rows


def compare_against_baselines(bed: Bed5, primary: Dict[str, Any],
                              baseline_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Paired significance tests, proposed vs each competitor, on identical rows.

    A paired test is available precisely *because* the competitors ran on the same
    manifest: for each test slice there are two predictions, so the discordant pairs
    are meaningful. That is not possible against a literature number, which is the
    substance of the first rejection ground rather than a technicality.
    """
    sp = primary.get("slice_predictions")
    if not sp:
        return []
    y_true = np.asarray(sp["y_true"])
    p_pred = np.asarray(sp["y_pred"])
    out: List[Dict[str, Any]] = []
    for r in baseline_rows:
        if r.get("status") != "ok" or "slice_predictions" not in r:
            continue
        b_pred = np.asarray(r["slice_predictions"]["y_pred"])
        if len(b_pred) != len(p_pred):
            continue
        mc = mcnemar(y_true, p_pred, b_pred)
        out.append(dict(
            bed=bed.key, comparison=f"proposed vs {r['label']}",
            competitor=r["key"], citation=r["citation"],
            proposed_acc=primary.get("test_acc"), competitor_acc=r.get("test_acc"),
            delta_acc=(float(primary.get("test_acc", 0.0)) - float(r.get("test_acc", 0.0))),
            proposed_bal_acc=primary.get("test_bal_acc"),
            competitor_bal_acc=r.get("test_bal_acc"),
            mcnemar_n01=mc["n01"], mcnemar_n10=mc["n10"], mcnemar_p=mc["p"],
            same_cohort=True, same_splits=True, same_preprocessing=True,
            manifest_digest=primary.get("manifest_digest"),
            note="paired exact McNemar on identical test rows; both models saw the same "
                 "manifest and the same sealed cohort"))
    return out

# ---------------------------------------------------------------------------
# Cross-validation stage
# ---------------------------------------------------------------------------
# Group-disjoint K-fold over train+valid only. This exists because the patient-wise
# bed is small -- 141 identifiers in train -- and a single validation split of that
# size gives a noisy estimate, so a selection made on it may not be reproducible.
# The test cohort is excluded by construction inside kfold_group_manifests, not by
# an argument someone could forget to pass.

def run_cv(cfg: Cfg5, bed: Bed5, store: FeatureStore, folds: int = 5,
           spec: Optional[str] = None, learner: Optional[str] = None,
           verbose: bool = True) -> List[Dict[str, Any]]:
    """Score one configuration over K group-disjoint folds of train+valid.

    Every fold refits standardization, PCA and the learner on that fold's training
    rows. Nothing is carried across folds, and nothing touches test. Returns one row
    per fold plus a summary row, so a reader sees the spread and not only the mean.
    """
    spec = spec or cfg.specs[0]
    learner = learner or (cfg.learners[0] if cfg.learners else "svm")
    folds_m = S.kfold_group_manifests(bed.split("train"), bed.split("valid"),
                                      k=int(folds), seed=cfg.random_state)
    M = bed.cost_matrix(cfg.c_miss, cfg.c_type, cfg.c_fp)
    X_pool, y_pool, pool_pos = _cv_pool(bed, store, spec, cfg)
    rows: List[Dict[str, Any]] = []

    for i, fm in enumerate(folds_m):
        tr_idx = _pool_index(fm["train"], pool_pos)
        va_idx = _pool_index(fm["valid"], pool_pos)
        X_tr, y_tr = X_pool[tr_idx], y_pool[tr_idx]
        X_va, y_va = X_pool[va_idx], y_pool[va_idx]
        rep, Z_tr = fit_representation(X_tr, cfg.standardize, cfg.pca_var,
                                       dims=spec_dims(spec), seed=cfg.random_state)
        Z_va = rep.transform(X_va)
        sw = None
        if cfg.use_hard_weighting:
            sw = confidence_weights(Z_tr, y_tr, bed.normal_idx, cfg.hard_weight_gamma,
                                    cfg.hard_weight_fn_penalty, seed=cfg.random_state)
        _p, proba, secs = fit_learner(learner, Z_tr, y_tr, Z_va, y_va, Z_va,
                                      bed.n_classes, cfg, sample_weight=sw)
        pred = proba.argmax(axis=1)
        row = dict(bed=bed.key, kind="cv_fold", fold=i, k=int(folds),
                   spec=spec, spec_label=spec_label(spec), learner=learner,
                   n_train=int(fm["train"].n), n_valid=int(fm["valid"].n),
                   n_train_groups=int(fm["train"].n_groups()),
                   n_valid_groups=int(fm["valid"].n_groups()),
                   group_unit=bed.group_unit, n_components=rep.n_components,
                   explained=rep.explained, fit_seconds=float(secs),
                   exp_risk=expected_risk(y_va, pred, M),
                   pool="train+valid", test_excluded=True)
        row.update(score_block(bed, y_va, pred))
        rows.append(row)
        if verbose:
            print(f"  fold {i}: acc={row['acc']*100:6.2f} bal={row['bal_acc']*100:6.2f} "
                  f"({row['n_train_groups']} train / {row['n_valid_groups']} valid "
                  f"{bed.group_unit}s, {secs:5.1f}s)")

    if rows:
        accs = np.array([r["acc"] for r in rows], dtype=np.float64)
        bals = np.array([r["bal_acc"] for r in rows], dtype=np.float64)
        summary = dict(bed=bed.key, kind="cv_summary", fold=-1, k=int(folds), spec=spec,
                       spec_label=spec_label(spec), learner=learner,
                       acc=float(accs.mean()), acc_std=float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
                       acc_min=float(accs.min()), acc_max=float(accs.max()),
                       bal_acc=float(bals.mean()),
                       bal_acc_std=float(bals.std(ddof=1)) if len(bals) > 1 else 0.0,
                       group_unit=bed.group_unit, pool="train+valid", test_excluded=True,
                       note="group-disjoint K-fold over train+valid; the sealed test "
                            "cohort is excluded by construction, not by convention")
        rows.append(summary)
        if verbose:
            print(f"  mean acc={summary['acc']*100:6.2f} +/- {summary['acc_std']*100:.2f}  "
                  f"bal={summary['bal_acc']*100:6.2f} +/- {summary['bal_acc_std']*100:.2f}")
    return rows


def _cv_pool(bed: Bed5, store: FeatureStore, spec: str,
             cfg: Cfg5) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Stack train and valid features once, and remember where each file landed.

    A fold's manifests name files, not rows of this stack, so the map is by absolute
    path. Building it once keeps K folds at one feature load instead of 2K.
    """
    X_tr, y_tr = store.spec_features("train", spec, 1)
    X_va, y_va = store.spec_features("valid", spec, cfg.tta)
    X = np.concatenate([X_tr, X_va], axis=0)
    y = np.concatenate([y_tr, y_va], axis=0)
    pos: Dict[str, int] = {}
    for i, f in enumerate(bed.split("train").abs_paths()):
        pos[f] = i
    off = len(y_tr)
    for i, f in enumerate(bed.split("valid").abs_paths()):
        pos[f] = off + i
    return X, y, pos


def _pool_index(m: "S.SplitManifest", pos: Dict[str, int]) -> np.ndarray:
    idx = []
    for f in m.abs_paths():
        if f not in pos:
            raise RuntimeError(f"fold manifest names {f}, which is in neither the train "
                               f"nor the valid pool. The folds were built from a "
                               f"different manifest than the features.")
        idx.append(pos[f])
    return np.asarray(idx, dtype=np.int64)

# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
# A run writes enough that someone who was not present can check the two claims the
# desk rejection doubted, without rerunning anything: the ledger says what was
# decided, on which split, in what order; the manifest digests say which rows those
# decisions were made on; the cache binds say which feature file each matrix came
# from. The CSVs are for reading, the JSON is the record.

def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    import csv
    if not rows:
        return ""
    skip = {"slice_predictions", "grid", "rule_scores", "rules_on_test"}
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys and k not in skip:
                keys.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _flat(r.get(k)) for k in keys})
    return path


def _flat(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return ";".join(str(_flat(x)) for x in v)
    if isinstance(v, dict):
        return json.dumps(S._jsonable(v), sort_keys=True)
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def write_artifact(cfg: Cfg5, bed: Bed5, ledger: "S.SelectionLedger",
                   store: FeatureStore, payload: Dict[str, Any],
                   name: str = "run") -> str:
    doc = dict(
        written_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        stage=payload.get("stage"),
        bed=bed.key, group_unit=bed.group_unit, class_names=list(bed.class_names),
        normal_index=bed.normal_idx,
        # Whether this cohort may carry a reported number, in the artifact rather
        # than only in the console log. The console scrolls away; this file is what
        # someone opens in a year to ask where Table N came from, and the answer
        # "that bed was demoted, here is the sentence saying why" has to be in it.
        **reportability(bed),
        config=cfg.to_json(), config_digest=cfg.digest(),
        splits={r: dict(n=bed.split(r).n, n_groups=bed.split(r).n_groups(),
                        class_counts=bed.split(r).class_counts(),
                        digest=bed.split(r).digest(),
                        recipe=bed.split(r).recipe) for r in S.ROLES},
        ledger=ledger.to_json(), features=store.report(),
        # The pixel path, in the file the manuscript's methods section is written
        # from. Rejection ground (1) names preprocessing as one of the conditions a
        # comparison has to hold constant, so it has to be recoverable from the run
        # record and not just from reading the source at some later commit.
        preprocessing=PRE.provenance(cfg.img_size,
                                     sorted({b for s in cfg.specs for b in s.split("+")}),
                                     tta=cfg.tta),
        environment=_environment(), payload=S._jsonable(payload))
    p = os.path.join(cfg.run_dir, f"{name}.json")
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(S._jsonable(doc), f, ensure_ascii=False, indent=2)
    return p


def reportability(bed: Bed5) -> Dict[str, Any]:
    """Columns/keys that say whether this bed's numbers may be quoted, and why not.

    Same keys the ablation harness stamps into its CSVs
    (ablation/common.reportability), so a row from runs_v5 and a row from
    ablation/out can be concatenated without one of them losing its licence. The
    two functions must return the SAME key set for that to hold, which is why the
    cohort-overlap columns are unpacked here too and not only there -- a run.json
    missing a column the CSVs carry is exactly the mismatch that makes a
    concatenation silently drop a caveat.
    """
    why = S.non_reportable_reason(bed.key)
    return dict(reportable=(not why), not_reportable_because=why or "",
                diagnostic_declared=S.diagnostic_run_declared() if why else "",
                **S.cohort_overlap(bed.key))


def guard_reportable(bed: Bed5, what: str) -> str:
    """Stop a table-producing stage on a demoted cohort, unless it was declared.

    Called at the top of select/report/baselines/cv -- the four stages whose
    output is a CSV somebody will later paste into the manuscript -- and before
    anything is frozen or read, so refusing costs nothing and leaves no ledger
    behind. `--stage dry-run` is deliberately not guarded: it fits nothing, reads
    nothing, and diagnosing a demoted bed's cache state is exactly what it is for.

    Raises S.NonReportableBed with the escape hatch spelled out. RUN_ALL.bat /diag
    sets the variable for the whole diagnostic block, so the retired tables stay
    reproducible in one command.
    """
    decl = S.assert_bed_reportable(bed.key, what)
    if decl:
        print(f"[diagnostic] bed {bed.key} is not reportable "
              f"({S.non_reportable_reason(bed.key)}).\n"
              f"             Allowed because: {decl}\n"
              f"             Every artifact this stage writes is stamped "
              f"reportable=False. Do not quote it.", flush=True)
    return decl


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = dict(python=sys.version.split()[0], numpy=np.__version__)
    for mod, key in (("sklearn", "sklearn"), ("torch", "torch"), ("xgboost", "xgboost"),
                     ("lightgbm", "lightgbm"), ("catboost", "catboost"),
                     ("torchvision", "torchvision"), ("scipy", "scipy")):
        try:
            env[key] = __import__(mod).__version__
        except Exception:
            env[key] = None
    try:
        import torch
        env["cuda"] = bool(torch.cuda.is_available())
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["cuda"] = None
        env["gpu"] = None
    return env

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def default_run_dir(bed: str) -> str:
    return os.path.join(HERE, "runs_v5", bed)


def make_cfg(bed: str, run_dir: Optional[str] = None, split_dir: Optional[str] = None,
             **over: Any) -> Cfg5:
    cfg = Cfg5(bed=bed, split_dir=split_dir or S.SPLIT_DIR,
               run_dir=run_dir or default_run_dir(bed))
    for k, v in over.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise AttributeError(f"Cfg5 has no field {k!r}")
        setattr(cfg, k, v)
    os.makedirs(cfg.run_dir, exist_ok=True)
    return cfg


def stage_dry_run(cfg: Cfg5) -> int:
    """Check everything a real run needs, and train nothing.

    Answers, in order: do the manifests exist; are the three roles disjoint at the
    grouping unit; does a feature matrix of the right shape exist or need extraction;
    and would the sealed cohort refuse a premature read. That last check is the one
    worth running before every experiment, because a protocol that is only tested
    when it succeeds has not been tested.
    """
    bed = load_bed(cfg.bed, cfg.split_dir, cfg.normal_class)
    print(f"bed {bed.key}  classes={list(bed.class_names)}  "
          f"normal_index={bed.normal_idx}  group_unit={bed.group_unit}")
    # Not guarded, but stated. A dry run on a demoted bed is a legitimate thing to
    # do -- it is how you check that bed's caches without touching anything -- so it
    # says what the later stages will refuse instead of refusing here.
    why = S.non_reportable_reason(bed.key)
    if why:
        print(f"NOT REPORTABLE: {why}.")
        print(f"  --stage select/report/baselines/cv will refuse this bed unless the run "
              f"declares itself diagnostic ({S.DIAGNOSTIC_ENV}=1, or RUN_ALL.bat /diag). "
              f"Reportable beds: {list(S.REPORTABLE_BEDS)}.")
    print(bed.manifest.summary())

    ov = S.group_overlap({r: bed.split(r) for r in S.ROLES})
    bad = {k: v for k, v in ov.items() if v}
    print(f"group overlap between roles: {'NONE' if not bad else bad}")
    if bad:
        print("  ^ this must be empty. A group in two roles means the same patient is "
              "in train and test, and every number downstream is optimistic.")

    store = FeatureStore(cfg, bed)
    print(f"cache dirs: {store.index.cache_dirs}")
    print(f"strict_tta={store.index.strict_tta}"
          f"{'' if store.index.strict_tta else '  (substitutions are served and recorded)'}")
    # The pixel path, stated and then checked. Stating it answers the preprocessing
    # half of rejection ground (1); checking it against v4's transform is what makes
    # the existing caches still valid, so the check belongs in the stage that runs
    # before an experiment rather than in a comment. A failure here is not fatal to a
    # dry run whose purpose is diagnosis, but it is reported as a failure.
    #
    # `ok` is initialised BEFORE the check, not after: setting it to True below would
    # have discarded a PIN BROKEN verdict and the dry run would have printed the
    # failure and then exited 0.
    ok = True
    print(f"\npreprocessing: {PRE.describe(cfg.img_size)}")
    # The measured channel content, printed here because this is the stage whose output
    # the preprocessing paragraph gets written from, and because the convenient claim
    # ("MRI is single-channel, so grayscale and RGB coincide") is true for figshare and
    # false for the four-class beds. Wrapped rather than one long line so it is read.
    import textwrap as _tw
    print("\n".join(_tw.wrap(PRE.channel_audit_note(), 96,
                             initial_indent="  ", subsequent_indent="  ")))
    # And which corpus this bed's pixels actually come from. Printed on every bed, not
    # only the pw ones, because the finding is symmetric and the mistake it prevents is
    # made while reading a table: figshare is 99.2% a mirrored subset of `balance
    # 4class`, so a pw number and a nick number are not two independent cohorts. The
    # per-bed exposure is already stamped into run.json by `reportability`; this is the
    # paragraph the cohorts subsection gets written from.
    if S.corpus_overlap_side(bed.key):
        print("\n".join(_tw.wrap(S.corpus_overlap_note(), 96,
                                 initial_indent="  ", subsequent_indent="  ")))
    # And what the TTA views do, from the same stage and for the same reason: eq. (4) is
    # written from this output, and "three views" does not say whether the averaging
    # probes pose or acquisition. Verified against v4's source first -- the table is a
    # description of code that lives in another file, which is the one kind of record
    # that can rot without anything failing.
    try:
        PRE.assert_tta_views_match_reference(verbose=True)
        print("\n".join(_tw.wrap(PRE.tta_note(int(cfg.tta)), 96,
                                 initial_indent="  ", subsequent_indent="  ")))
    except RuntimeError as e:
        ok = False
        print(f"  TTA VIEW RECORD STALE: {e}")
    try:
        PRE.assert_matches_reference(cfg.img_size, verbose=True)
    except ImportError as e:
        print(f"  pin unverified: {e} (constants are still pinned; needs torch)")
    except RuntimeError as e:
        ok = False
        print(f"  PIN BROKEN: {e}")
    for role in S.ROLES:
        m = bed.split(role)
        tta_n = 1 if role == "train" else cfg.tta
        for bb in sorted({b for s in cfg.specs for b in s.split("+")}):
            try:
                X, y = store.backbone_features(role, bb, tta_n)
                # What was asked for is not necessarily what was served, so print what
                # was served. A dry run that echoes the request tells you nothing about
                # the cache.
                eff = sorted({int(b["tta_effective"]) for b in store.binds
                              if b["role"] == role and b["backbone"] == bb})
                got = f"tta={tta_n}" if eff == [int(tta_n)] else \
                    f"tta={tta_n}->{','.join(str(e) for e in eff) or '?'}"
                print(f"  {role:5s} {bb:12s} {got:14s} {X.shape}  labels OK")
            except Exception as e:
                ok = False
                print(f"  {role:5s} {bb:12s} tta={tta_n}  UNAVAILABLE: "
                      f"{type(e).__name__}: {e}")

    subs = [b for b in store.binds if b.get("tta_substituted")]
    if subs:
        print(f"\nTTA substitutions: {len(subs)} bind(s) could not be served at the "
              f"requested setting.")
        for b in sorted({(x["role"], x["backbone"], x["tta_requested"],
                          x["tta_effective"], x["tta_note"]) for x in subs}):
            print(f"  {b[0]:5s} {b[1]:12s} asked tta={b[2]}, served tta={b[3]}")
            print(f"      {b[4]}")
        print("  These are recorded in every artifact, never relabelled. Pass "
              "--strict-tta to make them fail instead.")

    # The competitor backbones, checked separately and reported as a count rather than
    # as three more UNAVAILABLE lines per role. Their absence does not make a dry run
    # fail -- select and report do not need them -- but it does decide whether
    # `--stage baselines` can produce a complete same-condition table, which is the
    # answer to the first rejection ground. Better to learn that here than after the
    # selection grid has run.
    have, need, runnable = competitor_availability(cfg, bed, store)
    print(f"\ncompetitor backbones cached:  {', '.join(have) or 'none'}")
    print(f"competitor backbones missing: {', '.join(need) or 'none'}")
    for bb, why in need.items():
        print(f"    {bb:12s} {why}")
    print(f"baselines runnable now: {len(runnable)}/{len(BASELINES)} "
          f"({', '.join(sorted(runnable)) or 'none'})")
    if need:
        blocked = sorted(set(BASELINES) - set(runnable))
        n_img = sum(bed.split(r).n for r in S.ROLES)
        print(f"  blocked competitors: {', '.join(blocked)}")
        print(f"  --stage baselines will refuse to write a partial same-condition table "
              f"while any competitor is unavailable. Completing it costs one un-augmented "
              f"pass per missing backbone: {len(need)} x {n_img} = {len(need) * n_img} "
              f"forward passes (baseline_tta={cfg.baseline_tta}). Run once, on a GPU:\n"
              f"      python code_final_fixed_v5.py --bed {cfg.bed} --stage baselines "
              f"--extract-missing")

    led = S.SelectionLedger(os.path.join(cfg.run_dir, "_dryrun"), strict=True)
    try:
        led.authorize_test_read(cfg.bed, "dry-run probe")
        print("SEAL CHECK FAILED: an unfrozen ledger allowed a test read")
        ok = False
    except S.TestCohortViolation:
        print("seal check: an unfrozen ledger correctly refused a test read")
    print(f"\ndry run {'OK' if ok else 'INCOMPLETE -- see UNAVAILABLE lines above'}")
    return 0 if ok else 2


def stage_select(cfg: Cfg5, verbose: bool = True) -> int:
    bed = load_bed(cfg.bed, cfg.split_dir, cfg.normal_class)
    guard_reportable(bed, "freezing a selection")
    store = FeatureStore(cfg, bed)
    led = S.SelectionLedger(cfg.run_dir, strict=cfg.strict_test_seal)
    print(f"[select] bed={bed.key}  {len(cfg.specs)} specs x {len(cfg.learners)} learners "
          f"+ ensemble, scored on {bed.split('valid').n} validation rows")
    res = select_on_validation(cfg, bed, store, led, verbose=verbose)
    print("\n" + led.summary())
    save_selection(cfg, res)
    write_csv(os.path.join(cfg.run_dir, "selection_grid.csv"),
              [{k: v for k, v in r.items() if k != "rule_scores"} for r in res.grid])
    write_artifact(cfg, bed, led, store,
                   dict(stage="select", selection=res.to_json()), name="select")
    print(f"\nselected: {spec_label(res.spec)} + {LEARNER_LABEL.get(res.learner, res.learner)}, "
          f"rule={res.decision_rule}, T={res.temperature:.3f}, k={res.n_components}")
    print(f"sealed with digest {res.config_digest}. The test cohort is still unread.")
    print(f"artifacts in {cfg.run_dir}")
    return 0


def stage_report(cfg: Cfg5, verbose: bool = True) -> int:
    bed = load_bed(cfg.bed, cfg.split_dir, cfg.normal_class)
    # Before the ledger is even opened. A test read is irreversible, so the cheapest
    # possible moment to refuse is the right one: a demoted bed that gets this far
    # would spend its one read to produce a number nothing may quote.
    guard_reportable(bed, "reading the sealed test cohort")
    store = FeatureStore(cfg, bed)
    led = S.SelectionLedger.load(cfg.run_dir, strict=cfg.strict_test_seal)
    res = load_selection(cfg)
    if not led.frozen:
        raise S.TestCohortViolation(
            "the ledger in this run directory is not frozen. Run --stage select first.")
    print(f"[report] bed={bed.key}  sealed config {led.frozen_digest} frozen at {led.frozen_at}")
    print(f"         {spec_label(res.spec)} + {LEARNER_LABEL.get(res.learner, res.learner)}, "
          f"rule={res.decision_rule}")
    primary = report_on_test(cfg, bed, store, led, res, verbose=verbose)
    rows: List[Dict[str, Any]] = [dict(primary, kind="proposed")]

    print("\n[report] rule transfer, validation -> test")
    for r, v in sorted(primary["rules_on_test"].items()):
        mark = " <- selected on validation" if v["selected_on_valid"] else ""
        vb = v.get("valid_bal_acc")
        print(f"  {r:10s} test bal={v['bal_acc']*100:6.2f}  risk={v['exp_risk']:.4f}"
              + (f"  (valid bal={vb*100:6.2f})" if vb is not None else "") + mark)

    write_csv(os.path.join(cfg.run_dir, "results_test.csv"), rows)
    write_artifact(cfg, bed, led, store, dict(stage="report", primary=primary),
                   name="report")
    print("\n" + led.summary())
    print(f"artifacts in {cfg.run_dir}")
    return 0

def stage_baselines(cfg: Cfg5, keys: Optional[Sequence[str]] = None,
                    verbose: bool = True) -> int:
    """The same-condition comparison table.

    Runs after --stage report, and reuses that run's frozen ledger, so the
    competitors are evaluated on the identical sealed cohort with the identical
    manifests. The ledger's read counter grows by one per competitor, which is the
    honest accounting: the cohort was read once for the proposed method and once for
    each baseline, and the file says so.
    """
    bed = load_bed(cfg.bed, cfg.split_dir, cfg.normal_class)
    guard_reportable(bed, "writing a same-condition comparison table")
    store = FeatureStore(cfg, bed)
    led = S.SelectionLedger.load(cfg.run_dir, strict=cfg.strict_test_seal)
    res = load_selection(cfg)
    keys = list(keys) if keys else (list(cfg.baselines) if cfg.baselines else list(BASELINES))
    unknown = [k for k in keys if k not in BASELINES]
    if unknown:
        raise KeyError(f"unknown baseline(s) {unknown}; available: {sorted(BASELINES)}")

    print(f"[baselines] bed={bed.key}  {len(keys)} competitor(s) on manifest digest "
          f"{bed.split('test').digest()[:12]}, cohort n={bed.split('test').n}")

    # Availability is checked before the first competitor runs, not after the last one.
    # The reason is the ledger: a test read is permanent (SelectionLedger.load restores
    # the counter), so if this stage evaluated the two runnable competitors, raised, and
    # the user then extracted the missing backbones and re-ran, those two would raise
    # TestCohortViolation for having been read already -- and the complete table could
    # never be produced. Failing before any read keeps the retry clean.
    have, need, runnable = competitor_availability(cfg, bed, store, keys=keys)
    blocked = [k for k in keys if k not in runnable]
    if blocked and not (cfg.allow_partial_baselines or cfg.extract_missing):
        raise RuntimeError(
            f"{len(blocked)} of {len(keys)} competitors cannot run: "
            + ", ".join(blocked)
            + f"\nBackbones with no cache: {', '.join(need) or 'none'}"
            + "".join(f"\n    {bb:12s} {why}" for bb, why in need.items())
            + f"\n\nThe same-condition table is the answer to the first rejection ground, "
              f"so a partial one is not written by default -- and nothing has been read "
              f"from the sealed cohort yet, so either fix below and this stage runs clean."
              f"\n\nExtract the missing backbones once (one un-augmented pass each, "
              f"{len(need)} x {sum(bed.split(r).n for r in S.ROLES)} forward passes):\n"
              f"    python code_final_fixed_v5.py --bed {cfg.bed} --stage baselines "
              f"--extract-missing\n"
              f"or report a partial table and say in the paper which comparisons are "
              f"absent and why:\n"
              f"    python code_final_fixed_v5.py --bed {cfg.bed} --stage baselines "
              f"--allow-partial-baselines")
    if blocked:
        print(f"  {len(runnable)}/{len(keys)} competitors have cached features; "
              f"missing backbones: {', '.join(need) or 'none'}")

    rows = run_baselines(cfg, bed, store, led, res, keys=keys, verbose=verbose)

    # Second net, for failures that are not a missing cache: a head that would not fit,
    # a label mismatch, a corpus that changed size. Every skipped competitor is a
    # comparison the reviewer asked for and did not get.
    missing = [r for r in rows if r.get("status") != "ok"]
    if missing and not cfg.allow_partial_baselines:
        bbs = sorted({b for r in missing for b in BASELINES[r["key"]].spec.split("+")
                      if b not in BACKBONE_DIM})
        burned = [r["key"] for r in rows if r.get("test_cohort_read")]
        raise RuntimeError(
            f"{len(missing)} of {len(rows)} competitors could not be evaluated: "
            + ", ".join(f"{r['key']} ({r.get('error', 'unknown')[:60]})" for r in missing)
            + f"\n\nBackbones with no cache: "
              f"{bbs or 'none -- the cause is not a missing cache, read the errors above'}."
              f"\n\nEither extract them once:\n"
              f"    python code_final_fixed_v5.py --bed {cfg.bed} --stage baselines "
              f"--extract-missing\n"
              f"or, if you mean to report a partial table and say so in the paper, add "
              f"--allow-partial-baselines."
            + (f"\n\nNote: {len(burned)} competitor(s) already consumed a test read "
               f"({', '.join(burned)}), so a re-run needs --allow-repeat-test-reads and "
               f"the ledger will carry the warning."
               if burned else ""))
    if missing:
        print(f"  WARNING: writing a PARTIAL table -- {len(missing)} of {len(rows)} "
              f"competitors unavailable: {', '.join(r['key'] for r in missing)}. "
              f"The manuscript must say which comparisons are absent and why.")

    prior = os.path.join(cfg.run_dir, "report.json")
    comps: List[Dict[str, Any]] = []
    primary: Dict[str, Any] = {}
    if os.path.exists(prior):
        with open(prior, "r", encoding="utf-8") as f:
            primary = json.load(f).get("payload", {}).get("primary", {}) or {}
        comps = compare_against_baselines(bed, primary, rows)
    else:
        print("  note: report.json not found, so no paired comparison is written. "
              "Run --stage report first for the McNemar table.")

    table = ([dict(primary, kind="proposed", key="proposed",
                   label=f"{spec_label(res.spec)} + "
                         f"{LEARNER_LABEL.get(res.learner, res.learner)}",
                   citation="this work")] if primary else []) + rows
    write_csv(os.path.join(cfg.run_dir, "results_same_condition.csv"), table)
    if comps:
        write_csv(os.path.join(cfg.run_dir, "comparison_mcnemar.csv"), comps)
    write_artifact(cfg, bed, led, store,
                   dict(stage="baselines", baselines=rows, comparisons=comps),
                   name="baselines")

    print("\n  same-condition table (all rows: identical cohort, splits, preprocessing)")
    print(f"  {'method':44s} {'acc':>7s} {'bal':>7s} {'risk':>7s}  {'orig. claim':>11s}")
    if primary:
        print(f"  {'PROPOSED ' + spec_label(res.spec) + ' + ' + res.learner:44s} "
              f"{primary.get('test_acc', 0)*100:7.2f} {primary.get('test_bal_acc', 0)*100:7.2f} "
              f"{primary.get('test_exp_risk', 0):7.4f} {'--':>11s}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['label']:44s} {'SKIPPED':>7s}")
            continue
        rep = f"{r['reported_in_original']*100:.2f}%" if r.get("reported_in_original") else "n/a"
        print(f"  {r['label'] + ' (' + r['citation'] + ')':44s} "
              f"{r['test_acc']*100:7.2f} {r['test_bal_acc']*100:7.2f} "
              f"{r['test_exp_risk']:7.4f} {rep:>11s}")
    print("  the last column is what each paper reported on its own split. It is "
          "context, not a comparison -- only the measured columns are commensurable.")
    for c in comps:
        print(f"  {c['comparison']:52s} d_acc={c['delta_acc']*100:+6.2f}  "
              f"McNemar p={c['mcnemar_p']:.3g} (n01={c['mcnemar_n01']}, n10={c['mcnemar_n10']})")
    print(f"\nartifacts in {cfg.run_dir}")
    return 0


def stage_cv(cfg: Cfg5, folds: int = 5, spec: Optional[str] = None,
             learner: Optional[str] = None, verbose: bool = True) -> int:
    bed = load_bed(cfg.bed, cfg.split_dir, cfg.normal_class)
    guard_reportable(bed, "writing a cross-validation table")
    store = FeatureStore(cfg, bed)
    led = S.SelectionLedger(os.path.join(cfg.run_dir, "cv"), strict=cfg.strict_test_seal)
    print(f"[cv] bed={bed.key}  {folds}-fold, group-disjoint on {bed.group_unit}, "
          f"pool=train+valid ({bed.split('train').n + bed.split('valid').n} rows, "
          f"{bed.split('train').n_groups() + bed.split('valid').n_groups()} groups)")
    rows = run_cv(cfg, bed, store, folds=folds, spec=spec, learner=learner, verbose=verbose)
    write_csv(os.path.join(cfg.run_dir, f"cv_{folds}fold.csv"), rows)
    write_artifact(cfg, bed, led, store, dict(stage="cv", folds=folds, rows=rows),
                   name=f"cv_{folds}fold")
    print(f"artifacts in {cfg.run_dir}")
    return 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Frozen-embedding hybrid classifier, v5: sealed test cohort and "
                    "same-condition baselines.",
        epilog="Order matters: select, then report, then baselines. The ledger enforces "
               "it -- the reporting stage cannot run without a frozen selection, and "
               "the selection stage cannot see the test cohort.")
    ap.add_argument("--bed", default="nick",
                    choices=["nick", "nick_dedup", "bal", "imb", "pw",
                             "pw_identifier", "pw_family"])
    ap.add_argument("--stage", default="dry-run",
                    choices=["dry-run", "select", "report", "baselines", "cv", "all"])
    ap.add_argument("--split-dir", default=None,
                    help=f"manifest directory (default {S.SPLIT_DIR})")
    ap.add_argument("--run-dir", default=None, help="where artifacts go")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv-spec", default=None)
    ap.add_argument("--cv-learner", default=None)
    ap.add_argument("--baseline", action="append", default=None,
                    help="restrict to one competitor; repeatable. "
                         f"Available: {', '.join(sorted(BASELINES))}")
    ap.add_argument("--specs", default=None,
                    help="comma-separated fusion specs to search (default: all seven)")
    ap.add_argument("--learners", default=None, help="comma-separated subset of svm,xgb,lgb,cat")
    ap.add_argument("--pca-var", type=float, default=None)
    ap.add_argument("--tta", type=int, default=None)
    ap.add_argument("--decision-rule", default=None,
                    choices=["auto", "argmax", "threshold", "bayes", "bayes_cal"])
    ap.add_argument("--bootstrap-n", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--refit-on-train-valid", action="store_true",
                    help="refit the sealed configuration on train+valid before the test "
                         "read. Legitimate, but it changes what the number means, so it "
                         "is off by default and always recorded.")
    ap.add_argument("--no-cluster-bootstrap", action="store_true",
                    help="disable the patient-clustered interval (not recommended: on a "
                         "bed whose slices share patients the i.i.d. interval is too narrow)")
    ap.add_argument("--allow-repeat-test-reads", action="store_true",
                    help="downgrade a second read of the same cohort for the same purpose "
                         "from an error to a recorded warning")
    ap.add_argument("--extract", action="store_true",
                    help="allow fresh feature extraction instead of binding to the "
                         "existing caches (needs torch and a GPU to be quick). This "
                         "re-extracts everything, including the three backbones whose "
                         "caches produced the submitted numbers; --extract-missing is "
                         "almost always the flag you want instead.")
    ap.add_argument("--extract-missing", action="store_true",
                    help="bind to a cache wherever one exists and extract only the "
                         "backbones that have none. Required for the competitor "
                         "backbones (googlenet, vgg19, densenet169, resnext50), which "
                         "were never extracted, so without it --stage baselines can only "
                         "report 2 of the 7 registered competitors.")
    ap.add_argument("--allow-partial-baselines", action="store_true",
                    help="write the same-condition table even if some competitors could "
                         "not be run. Off by default: a table missing the closest "
                         "published analogue does not answer the comparability question, "
                         "and one that looks complete is worse than one that fails.")
    ap.add_argument("--strict-tta", action="store_true",
                    help="refuse to run when the requested TTA setting has no cache, "
                         "instead of serving the setting that does exist and recording "
                         "the substitution. The one substitution that happens on every "
                         "four-class run is tta=3 on the validation side, whose rows come "
                         "from the never-augmented Train cache. Equivalent to "
                         "ABLATION_STRICT_TTA=1.")
    ap.add_argument("--allow-non-reportable", action="store_true",
                    help="run a stage on bal or imb, whose cohorts no published work "
                         "uses (see splits_v5.NON_REPORTABLE_BEDS). Off by default: "
                         "those beds cannot answer the comparability question, so a "
                         "table from them is a trap for whoever reads the CSV later. "
                         "With this flag every artifact is stamped reportable=False "
                         f"and the reason travels in it. Same as {S.DIAGNOSTIC_ENV}=1.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)

    if a.allow_non_reportable:
        S.declare_diagnostic_run(f"--allow-non-reportable on the command line "
                                 f"(bed {a.bed}, stage {a.stage})")

    cfg = make_cfg(
        a.bed, run_dir=a.run_dir, split_dir=a.split_dir,
        img_size=a.img_size, pca_var=a.pca_var, tta=a.tta,
        decision_rule=a.decision_rule, bootstrap_n=a.bootstrap_n,
        specs=tuple(s.strip() for s in a.specs.split(",")) if a.specs else None,
        learners=tuple(s.strip() for s in a.learners.split(",")) if a.learners else None,
        refit_on_train_valid=True if a.refit_on_train_valid else None,
        cluster_bootstrap=False if a.no_cluster_bootstrap else None,
        strict_test_seal=False if a.allow_repeat_test_reads else None,
        strict_tta=True if a.strict_tta else None,
        prefer_existing_caches=False if a.extract else None,
        extract_missing=True if a.extract_missing else None,
        allow_partial_baselines=True if a.allow_partial_baselines else None,
        baselines=tuple(a.baseline) if a.baseline else None)
    verbose = not a.quiet

    if not os.path.isdir(cfg.split_dir):
        print(f"no manifests at {cfg.split_dir}.\nRun this first:  python splits_v5.py --build")
        return 2

    try:
        if a.stage == "dry-run":
            return stage_dry_run(cfg)
        if a.stage == "select":
            return stage_select(cfg, verbose=verbose)
        if a.stage == "report":
            return stage_report(cfg, verbose=verbose)
        if a.stage == "baselines":
            return stage_baselines(cfg, keys=a.baseline, verbose=verbose)
        if a.stage == "cv":
            return stage_cv(cfg, folds=a.folds, spec=a.cv_spec, learner=a.cv_learner,
                            verbose=verbose)
        if a.stage == "all":
            rc = stage_select(cfg, verbose=verbose)
            if rc:
                return rc
            rc = stage_report(cfg, verbose=verbose)
            if rc:
                return rc
            return stage_baselines(cfg, keys=a.baseline, verbose=verbose)
    except S.TestCohortViolation as e:
        print(f"\nTEST COHORT VIOLATION\n{e}")
        return 3
    except S.NonReportableBed as e:
        # Exit 4, distinct from 3: nothing went wrong with the protocol here and
        # nothing has to be undone. The bed was simply the wrong one, and
        # RUN_ALL.bat can tell the two apart without parsing the message.
        print(f"\nNON-REPORTABLE BED\n{e}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
