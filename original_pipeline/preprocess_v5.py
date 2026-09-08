"""preprocess_v5.py -- the pixel path, named instead of defaulted.

Rejection ground (1) is about comparability: "the same cohorts, training and
validation splits, PREPROCESSING, evaluation criteria, and held-out test cases".
A preprocessing step that is whatever the installed torchvision happens to default
to is not a stated condition, it is an accident that currently holds. Today
`T.Resize((224, 224))` means bilinear-with-antialias because that is torchvision
0.20.1's default; the antialias default flipped from False to True in 0.17, and a
reader who pip-installs a different version reproduces different embeddings from
identical code and identical files.

So every argument is written out here, and the values written out are exactly what
the reference implementation (code_final_fixed_v4.base_preprocess_transform) does
today -- verified by assert_matches_reference(), not by inspection. This module is a
pin, not a change: if it altered a single pixel it would invalidate the caches in
ablation/caches that produced the submitted numbers.

What the pin says, and what it deliberately does not claim:

  * The resize is SQUARE -- both sides go to img_size, aspect ratio is not preserved
    and nothing is cropped. This is a deviation from the torchvision preset the
    ImageNet weights were validated with (shorter side to 256, then a 224 centre
    crop), and it is a deliberate one: a centre crop discards the periphery of an
    axial slice, where meningiomas -- extra-axial by definition -- live. The cost is
    an anisotropic rescale of non-square slices. Stated, not hidden.
  * Antialiasing is on. clean-fid (arXiv 2104.11222) is the only measured evidence
    that bears on frozen feature extraction: resize implementations that skip the
    low-pass step shift the features enough to move a published metric by more than
    the effects people report. That finding is about FID, not accuracy, so it is a
    reason to pin the resize rather than evidence that a particular choice wins.
  * Normalization is per backbone by construction, ImageNet-canonical in fact. All
    six registered backbones ship torchvision presets built from
    `ImageClassification`'s default mean/std, so one constant is correct for all of
    them -- but it is correct by coincidence of the weight metadata, not by rule, so
    the mapping is explicit and an unregistered backbone raises.
  * Grayscale is not converted anywhere. The loader does `Image.open(p).convert(
    "RGB")` and this module keeps that. The tempting sentence -- "MRI is a
    single-channel modality, so grayscale loading and RGB loading coincide" -- was
    measured by channel_audit_v5.py over every file, and it is TRUE for figshare and
    FALSE for the four-class corpus:
      - figshare: 3064/3064 images are PIL mode "L", R==G==B in 100.00%.
      - four-class (`bal`/`nick`/`nick_dedup`, and `imb` which draws from the same
        files): 7200 images, modes RGB 4129 / L 3067 / RGBA 3 / P 1, R==G==B in
        7071/7200 = 98.21%, worst channel spread 214 of 255 levels.
    So for the reported four-class beds this IS a choice, and `convert("RGB")` is the
    conservative side of it: it passes the 129 colour-bearing files through unchanged,
    whereas a grayscale conversion would mix their channels. That matters more than
    1.79% suggests, because all 129 sit in ONE class -- `notumor`, at 38/400 of Test
    and 91/1400 of Train -- so a grayscale step would alter one class's pixels and no
    other's. The colour is content, not codec noise (PACS screenshots and pseudo-colour
    overlays; sample RGB triples (0,41,99) and (116,157,209)), which also makes it a
    candidate shortcut in its own right: a model can read "coloured => notumor" without
    looking at anatomy. Recorded here, not silently fixed -- removing or graying those
    files would change the published cohort, which is the comparability claim this
    repair exists to protect. The ablation that belongs in the manuscript is
    grayscale-vs-RGB on the deduplicated bed, not a change of default.
  * The TTA views are part of this pixel path, not a setting downstream of it.
    FeatureStore.extract does `pre(vf(im))`, so v4's tta_variants runs on the PIL image
    BEFORE the pinned resize. Measured (TTA_VIEWS / TTA_WEDGE_AUDIT below): the
    protocol default tta=3 is {identity, hflip, +7deg rotation} -- identity is in the
    set, but the set is purely geometric, and the rotation defaults to NEAREST
    resampling with black fill, so it destroys 5.41% of the frame before the resize
    this module is careful about ever runs. Recorded, not changed: a different view set
    is a different experiment.

Nothing here imports torch at module level: splits can be built and checked on a
machine with no torch installed, and that property is load-bearing for the static
checkers.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

# ---------------------------------------------------------------------------
# What the corpora actually contain, measured
# ---------------------------------------------------------------------------
# channel_audit_v5.py, every file, judged on what `convert("RGB")` hands the model
# rather than on the file's declared mode (a palette image decodes to 2-D indices and
# would otherwise be miscounted as single-channel). Transcribed here so the artifact
# records the condition instead of only the audit's console output doing so, and so the
# manuscript cannot state the equivalence for a corpus where it does not hold.
#
# These are measurements, not settings: nothing in the pixel path reads them, and
# editing one would not change a single pixel -- it would only make the artifact lie.
CHANNEL_AUDIT: Dict[str, Dict[str, Any]] = {
    "figshare": {
        "n_images": 3064, "n_rgb_identical": 3064, "frac_rgb_identical": 1.0,
        "modes": {"L": 3064}, "worst_channel_spread": 0,
        "single_channel_in_effect": True,
        "colour_bearing_classes": [],
    },
    "four_class": {
        # One row covers every four-class bed because they all read the same tree:
        # `bal`, `nick` and `nick_dedup` point at `balance 4class` directly, and the
        # `imb` bed is an RNG subsample of that tree's train split rather than a
        # separate corpus. (The `imbanlce 4class` folder on disk is a truncated copy of
        # the same files -- 5400 of them, notumor kept whole -- and no bed builder reads
        # it, so it gets no row of its own either.)
        "n_images": 7200, "n_rgb_identical": 7071,
        "frac_rgb_identical": 7071 / 7200,
        "modes": {"RGB": 4129, "L": 3067, "RGBA": 3, "P": 1},
        "worst_channel_spread": 214,
        "single_channel_in_effect": False,
        # The number that matters is not 1.79% but that it is confined to one class.
        "colour_bearing_classes": ["notumor"],
        "colour_bearing_counts": {"Test/notumor": "38/400", "Train/notumor": "91/1400"},
        # Of the 129, most are faint (median spread 11 of 255); these are the ones that
        # are unmistakably colour rather than a codec artefact. The threshold is stated
        # because the count is insensitive to it: >20 levels over >0.1% of pixels gives
        # 33, and so does >1%, while dropping the area condition entirely gives 36.
        "n_strong_colour": 33,
        "strong_colour_rule": ">20 levels of channel spread over >0.1% of pixels",
        "spread_min_median_max": [2, 11, 214],
        "colour_bearing_modes": {"RGB": 126, "RGBA": 2, "P": 1},
    },
    "audited_by": "channel_audit_v5.py (2026-09-05, all files, no sampling)",
}

# ---------------------------------------------------------------------------
# What the TTA views actually do, measured
# ---------------------------------------------------------------------------
# Eq. (4) averages embeddings over `tta` views. The manuscript states the count. This
# states the views, because the count alone cannot distinguish an intensity-based view
# set (which leaves geometry alone and probes robustness to contrast and noise) from a
# geometric one (which moves pixels and, without expand, deletes some of them).
#
# The view set is v4's `tta_variants(n)`, taken in order, so the protocol default n=3 is
# exactly TTA_VIEWS[:3]. It is applied as `pre(vf(im))` in FeatureStore.extract -- on the
# PIL image, BEFORE the pinned resize -- which is why it belongs in this module and not
# only in the runner: whatever the wedge does, the resize this module pins then
# propagates it.
#
# Like CHANNEL_AUDIT, these are measurements and one declaration, never settings. No
# tensor, rule id or cache key reads them (asserted), so correcting a number here cannot
# invalidate a cached matrix. The view set itself lives in code_final_fixed_v4.py and is
# NOT redefined here -- a second copy would be a second source of truth, and the
# hazard being recorded is precisely that the two could drift apart.
TTA_VIEWS: Tuple[Dict[str, Any], ...] = (
    {"name": "id", "kind": "identity", "geometric": False,
     "note": "the unmodified image; present, so tta>1 can only dilute the clean view, "
             "never replace it"},
    {"name": "hflip", "kind": "reflection", "geometric": True,
     "note": "exact, lossless, no interpolation and no fill. But a brain is only "
             "approximately mirror-symmetric and lateralization is diagnostic, so this "
             "is not a null transform: measured mean |delta| is 21.6 grey levels "
             "(median over 80 slices), against 42.0 levels between two DIFFERENT "
             "images -- i.e. about half the distance to an unrelated slice"},
    {"name": "rot_p7", "kind": "rotation", "geometric": True,
     "note": "TF.rotate(im, 7) with every argument defaulted: NEAREST resampling "
             "(no interpolation of intensities, so edges alias) and fill=None (black). "
             "expand=False, so the frame is not enlarged and the corners are cut off"},
    {"name": "rot_m7", "kind": "rotation", "geometric": True,
     "note": "same as rot_p7 at -7 degrees; only reached at tta>=4, which no reported "
             "configuration uses"},
    {"name": "zoom", "kind": "crop+rescale", "geometric": True,
     "note": "centre crop to 92% then resize back, via PIL's `Image.resize` default "
             "(BICUBIC) rather than the pinned bilinear+antialias path -- a second, "
             "unpinned resize. Only reached at tta>=5"},
)

# The 7-degree rotation without expand=True cuts the frame. Two separate quantities,
# because they answer different questions and the smaller one is the honest headline:
#   * frame fraction: pure geometry, scale-invariant, identical for every image;
#   * foreground fraction: how much of the *brain* (>10/255) it actually removes.
# Measured over every file of both corpora, no sampling.
TTA_WEDGE_AUDIT: Dict[str, Any] = {
    "angle_deg": 7, "resample": "nearest", "fill": "black (0)", "expand": False,
    "frame_fraction_blanked": 0.0541,
    "frame_fraction_note": "scale-invariant: 5.41% at 224x224 and at 512x512 alike, and "
                           "up to 86% of a 32x32 corner block at 512x512",
    "foreground_threshold": "intensity > 10 of 255",
    "four_class": {
        "n_images": 7200, "pooled_foreground_lost": 0.0103,
        "per_class_pooled": {"glioma": 0.0039, "meningioma": 0.0072,
                             "notumor": 0.0227, "pituitary": 0.0114},
        # The dispersion matters more than the mean: a per-class average of 2% hides
        # 267 notumor images that lose more than a twentieth of their brain.
        "n_losing_over_5pct": 319,
        "n_losing_over_5pct_per_class": {"glioma": 10, "meningioma": 38,
                                         "notumor": 267, "pituitary": 4},
        "n_losing_nothing": 2015,
        "worst_per_image_loss": 0.0697,
    },
    "figshare": {
        "n_images": 3064, "pooled_foreground_lost": 0.0075,
        "median_per_image_loss": 0.0043, "p90_per_image_loss": 0.0159,
        "n_losing_over_5pct": 0, "worst_per_image_loss": 0.0315,
        "note": "uniformly framed 512x512 slices with a wide dark margin, so the wedge "
                "lands almost entirely on background -- no image loses 5% of its "
                "foreground. The four-class corpus has 447 distinct sizes and tightly "
                "cropped PACS screenshots, which is why the same rotation behaves worse "
                "there. The view set is a per-corpus hazard, not a constant one",
    },
    "audited_by": "measured 2026-09-05 over every file of both corpora (no sampling); "
                  "PIL Image.rotate(7, NEAREST, expand=False, fillcolor=None), which is "
                  "what torchvision's PIL branch calls",
}


def tta_view_names(n: int = 3) -> Tuple[str, ...]:
    """The views a given tta setting uses, in order. Mirrors v4's tta_variants slicing."""
    return tuple(v["name"] for v in TTA_VIEWS[:max(1, int(n))])


def tta_note(n: int = 3) -> str:
    """One paragraph for --stage dry-run, the artifact, and the manuscript's eq. (4)."""
    used = TTA_VIEWS[:max(1, int(n))]
    fc, fs = TTA_WEDGE_AUDIT["four_class"], TTA_WEDGE_AUDIT["figshare"]
    geo = [v["name"] for v in used if v["geometric"]]
    has_id = any(not v["geometric"] for v in used)
    return (
        f"TTA at tta={int(n)} averages the embeddings of {len(used)} views: "
        f"{', '.join(v['name'] for v in used)}. Identity "
        f"{'IS' if has_id else 'is NOT'} among them, so the clean view is "
        f"{'diluted, not replaced' if has_id else 'absent'}. Every other view is "
        f"geometric ({', '.join(geo)}) -- there is no intensity-based view (no noise, "
        f"blur, gamma or contrast perturbation) anywhere in the set, so eq. (4) probes "
        f"invariance to pose and not to acquisition. The rotation is the costly one: "
        f"TF.rotate defaults to NEAREST resampling with black fill and expand=False, so "
        f"it blanks {100 * TTA_WEDGE_AUDIT['frame_fraction_blanked']:.2f}% of the frame "
        f"at every scale, ahead of the pinned resize. On the four-class corpus that "
        f"removes {100 * fc['pooled_foreground_lost']:.2f}% of foreground pixels "
        f"overall but is unevenly distributed -- "
        f"{100 * fc['per_class_pooled']['notumor']:.2f}% of notumor against "
        f"{100 * fc['per_class_pooled']['glioma']:.2f}% of glioma, and "
        f"{fc['n_losing_over_5pct']} images lose more than 5% of their brain, "
        f"{fc['n_losing_over_5pct_per_class']['notumor']} of them notumor. On figshare "
        f"the same rotation costs {100 * fs['pooled_foreground_lost']:.2f}% and no "
        f"single image loses 5%, because those slices are uniformly framed with a wide "
        f"dark margin. Reported as measured; the view set is not changed here, because "
        f"a different view set is a different experiment and the caches behind the "
        f"submitted numbers hold this one.")


def channel_audit_note() -> str:
    """One sentence per corpus, for --stage dry-run and for the artifact's notes."""
    fs, fc = CHANNEL_AUDIT["figshare"], CHANNEL_AUDIT["four_class"]
    return (
        f"measured channel content ({CHANNEL_AUDIT['audited_by']}): figshare is "
        f"{fs['n_rgb_identical']}/{fs['n_images']} R==G==B (100%), so RGB loading and "
        f"grayscale-then-triplicate are the same tensor there; the four-class corpus is "
        f"{fc['n_rgb_identical']}/{fc['n_images']} "
        f"({100.0 * fc['frac_rgb_identical']:.2f}%), worst channel spread "
        f"{fc['worst_channel_spread']}/255, and all "
        f"{fc['n_images'] - fc['n_rgb_identical']} colour-bearing files are in "
        f"{fc['colour_bearing_classes'][0]!r} alone "
        f"({', '.join(f'{k} {v}' for k, v in fc['colour_bearing_counts'].items())}). "
        f"convert('RGB') is therefore the conservative path on the four-class beds -- it "
        f"leaves those files unchanged, where a grayscale step would alter the pixels of "
        f"one class and no other. Class-exclusive colour is itself a candidate shortcut "
        f"and is reported as such, not removed: deleting or graying those files would "
        f"change the published cohort.")


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------
# Names, not numbers, because torchvision spells these as enum members and a bare
# integer would be the same unnamed-default problem one level down.
RESIZE_GEOMETRY = "square"          # both sides -> img_size; no crop, no aspect keep
RESIZE_INTERPOLATION = "bilinear"   # InterpolationMode.BILINEAR
RESIZE_ANTIALIAS = True             # explicit; the 0.17 default flip is why
CHANNELS = "rgb"                    # PIL .convert("RGB"); triplicates 1-channel input

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Per-backbone normalization. Every entry is the ImageNet constant because every
# registered backbone's torchvision preset uses ImageClassification's default
# mean/std -- checked against the installed weight metadata, not assumed. The dict
# exists so that adding a backbone whose preset differs (any SWAG ViT: bicubic,
# 384 px, and its own constants) fails at build time instead of being normalized
# with the wrong numbers and producing plausible-looking features.
NORM_BY_BACKBONE: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    # the proposed method
    "resnet18": (IMAGENET_MEAN, IMAGENET_STD),
    "densenet121": (IMAGENET_MEAN, IMAGENET_STD),
    "vit_b_16": (IMAGENET_MEAN, IMAGENET_STD),
    # the same-condition competitor set
    "googlenet": (IMAGENET_MEAN, IMAGENET_STD),
    "vgg19": (IMAGENET_MEAN, IMAGENET_STD),
    "densenet169": (IMAGENET_MEAN, IMAGENET_STD),
    "resnext50": (IMAGENET_MEAN, IMAGENET_STD),
    # reachable through v4's build_extractor
    "resnet50": (IMAGENET_MEAN, IMAGENET_STD),
    "efficientnet_b0": (IMAGENET_MEAN, IMAGENET_STD),
}

# Aliases the extractors already accept. Kept beside the table rather than inside
# it so the table stays a statement about models, one row per set of weights.
_ALIAS = {"vit": "vit_b_16", "vitb16": "vit_b_16", "densenet": "densenet121",
          "d121": "densenet121", "d169": "densenet169",
          "resnext50_32x4d": "resnext50", "vgg_19": "vgg19",
          "inception_v1": "googlenet", "efficientnet": "efficientnet_b0",
          "effb0": "efficientnet_b0",
          "xrv": "xraytorchvision", "xrv_densenet": "xraytorchvision"}

# Backbones the extractor can build but this pin refuses to serve, with the reason.
# Separated from a plain absence because the two need different answers: an absent
# backbone should be looked up and added, whereas these are wrong here on purpose and
# adding them would be the mistake.
_UNSUPPORTED = {
    "xraytorchvision":
        "torchxrayvision's DenseNet does its own scaling inside the extractor "
        "(x.mean(dim=1) * 2048 - 1024, single channel, [-1024, 1024] Hounsfield-like "
        "units). ImageNet mean/std in front of that would be scaled twice. It is not "
        "used by any spec or baseline in this project; if it ever is, the pin needs a "
        "second pixel path, not an entry in this table.",
}


def canonical_backbone(backbone: str) -> str:
    b = str(backbone).lower().strip()
    return _ALIAS.get(b, b)


def norm_for(backbone: str) -> Tuple[Tuple[float, float, float],
                                    Tuple[float, float, float]]:
    """Mean/std for `backbone`. Raises for anything unregistered.

    Deliberately not `.get(b, (IMAGENET_MEAN, IMAGENET_STD))`: a default here would
    normalize an unknown backbone with constants that may not be its own, and the
    features would look entirely reasonable while meaning something else.

    Two different failures, two different messages. An unknown name is an omission
    and the fix is to look up its preset and add a row. A name in _UNSUPPORTED is a
    refusal on purpose, and the fix is emphatically NOT to add a row -- so it says so,
    rather than pointing the reader at the table and inviting the wrong repair.
    """
    b = canonical_backbone(backbone)
    if b in _UNSUPPORTED:
        raise KeyError(f"backbone {backbone!r} is deliberately not pinned here: "
                       f"{_UNSUPPORTED[b]}")
    if b not in NORM_BY_BACKBONE:
        raise KeyError(
            f"no pinned normalization for backbone {backbone!r}. Add it to "
            f"NORM_BY_BACKBONE with the constants from its own torchvision preset "
            f"(weights.transforms()), after checking they are the ImageNet defaults "
            f"-- the SWAG ViT presets, for one, are not. Registered: "
            f"{sorted(NORM_BY_BACKBONE)}")
    return NORM_BY_BACKBONE[b]


def preprocess_rule_id(img_size: int, backbone: str) -> str:
    """Identity of the pixel path, for cache keys and for the artifact.

    Everything that changes a pixel is in here and nothing that does not. It goes
    into the v5 cache key so that a future change to any of it cannot silently reuse
    embeddings extracted under the old one -- the same reason img_size and tta are
    already in that key.
    """
    mean, std = norm_for(backbone)
    m = ",".join(f"{v:g}" for v in mean)
    s = ",".join(f"{v:g}" for v in std)
    aa = "aa" if RESIZE_ANTIALIAS else "noaa"
    return (f"{RESIZE_GEOMETRY}{int(img_size)}-{RESIZE_INTERPOLATION}-{aa}"
            f"|{CHANNELS}|norm({m};{s})")


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------

def build_transform(img_size: int = 224, backbone: str = "resnet18"):
    """The pinned pixel path as a torchvision transform.

    Same three steps as v4's base_preprocess_transform, with every default written
    out. assert_matches_reference() proves the two agree bit for bit; this is the
    version that keeps agreeing when torchvision's defaults move.

    torch is imported here, not at module scope, so that importing this module to
    read the constants costs nothing and works without torch installed.
    """
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    interp = {"bilinear": InterpolationMode.BILINEAR,
              "bicubic": InterpolationMode.BICUBIC,
              "nearest": InterpolationMode.NEAREST}[RESIZE_INTERPOLATION]
    mean, std = norm_for(backbone)
    s = int(img_size)
    return T.Compose([
        # (s, s) and not s: a bare int resizes the SHORTER side to s and keeps the
        # aspect ratio, which for a 512x384 slice would give 298x224 and then fail to
        # stack into a batch. The square form is also the reference implementation's,
        # so this is the pin, not a preference.
        T.Resize((s, s), interpolation=interp, antialias=RESIZE_ANTIALIAS),
        T.ToTensor(),                       # uint8 [0,255] HWC -> float [0,1] CHW
        T.Normalize(mean=list(mean), std=list(std)),
    ])


def describe(img_size: int = 224, backbone: str = "resnet18") -> str:
    """One line for the artifact and for --stage dry-run."""
    mean, std = norm_for(backbone)
    return (f"resize {RESIZE_GEOMETRY} ({img_size}x{img_size}, {RESIZE_INTERPOLATION}, "
            f"antialias={RESIZE_ANTIALIAS}, no crop, aspect ratio not preserved) -> "
            f"{CHANNELS} -> ToTensor -> Normalize(mean={list(mean)}, std={list(std)}) "
            f"[{canonical_backbone(backbone)}]")


def provenance(img_size: int = 224,
               backbones: Sequence[str] = ("resnet18", "densenet121", "vit_b_16"),
               tta: int = 3) -> Dict[str, Any]:
    """What the artifact records, so the manuscript can state it without guessing."""
    return {
        "rule_id": {b: preprocess_rule_id(img_size, b) for b in backbones},
        "img_size": int(img_size),
        "resize_geometry": RESIZE_GEOMETRY,
        "interpolation": RESIZE_INTERPOLATION,
        "antialias": bool(RESIZE_ANTIALIAS),
        "center_crop": False,
        "aspect_ratio_preserved": False,
        "channels": CHANNELS,
        "normalization": {canonical_backbone(b): {"mean": list(norm_for(b)[0]),
                                                  "std": list(norm_for(b)[1])}
                          for b in backbones},
        "identical_across_splits": True,
        # Not a setting: what the files contain. It rides in the artifact because the
        # reviewer's question ("why a three-channel normalization for MRI?") is answered
        # by the corpus, not by the transform, and an artifact that recorded only the
        # transform would leave the answer nowhere.
        "channel_audit": CHANNEL_AUDIT,
        # The views eq. (4) averages, and what the rotation costs. Same reason as the
        # channel audit: the count `tta=3` is already in the config, but "three views"
        # does not say whether the augmentation moved pixels or only their intensities,
        # and the reviewer's comparability question is about the condition, not the
        # cardinality. `tta_requested` and not `tta_effective` -- what a cache was
        # actually served at is the runner's bind log to report, and duplicating it here
        # would create a second answer to one question.
        "tta": {
            "requested": int(tta),
            "views": list(tta_view_names(tta)),
            "view_catalog": [dict(v) for v in TTA_VIEWS],
            "includes_identity": "id" in tta_view_names(tta),
            "all_non_identity_views_geometric": True,
            "intensity_based_views": [],
            "applied": "to the PIL image before the pinned resize (pre(vf(im)))",
            "wedge_audit": TTA_WEDGE_AUDIT,
        },
        "notes": [
            "one transform object serves train, valid and test; there is no "
            "train-only augmentation in the pixel path. TTA is applied at inference "
            "only, in front of this transform, and is recorded under 'tta' with the "
            "views it uses -- not only their count",
            "square resize rather than the torchvision preset's resize-256 + "
            "center-crop-224: a centre crop removes the periphery of an axial slice, "
            "where extra-axial meningiomas sit. The cost is an anisotropic rescale of "
            "non-square slices",
            "every argument is explicit because torchvision's antialias default "
            "changed in 0.17; identical code on a different version would otherwise "
            "produce different embeddings from identical files",
            channel_audit_note(),
            tta_note(tta),
        ],
    }


def assert_matches_reference(img_size: int = 224, backbone: str = "resnet18",
                            verbose: bool = True) -> Dict[str, Any]:
    """Prove the pin changes nothing: same tensor as v4, to the last bit.

    This is the only thing that makes the pin safe to adopt mid-project. The caches
    in ablation/caches were extracted with v4's transform and produced the submitted
    numbers; if this function ever fails, the pin has become a change and every
    cached embedding is stale. Compared with array_equal and not allclose -- a
    tolerance would hide exactly the sub-pixel differences that resize changes cause.
    """
    import importlib.util
    import os

    import numpy as np
    import torch
    from PIL import Image

    here = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(here, "code_final_fixed_v4.py")
    spec = importlib.util.spec_from_file_location("base_v4_pre", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    ours = build_transform(img_size, backbone)
    theirs = base.base_preprocess_transform(img_size)

    rng = np.random.default_rng(0)
    # Non-square and odd sizes on purpose: a square 224 input would pass under any
    # resize rule and prove nothing. 512x512 is the corpus's most common size, 512x384
    # exercises the aspect-ratio decision, 224x224 is the identity case, and 137x211
    # is an upscale from prime-ish sides where interpolation differences are largest.
    cases, out = [(512, 512), (512, 384), (224, 224), (137, 211)], []
    for w, h in cases:
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        im = Image.fromarray(arr, "RGB")
        a, b = ours(im), theirs(im)
        same = bool(torch.equal(a, b))
        md = float((a - b).abs().max()) if a.shape == b.shape else float("nan")
        out.append(dict(size=f"{w}x{h}", identical=same, max_abs_diff=md,
                        shape=tuple(a.shape)))
        if verbose:
            print(f"  {'OK  ' if same else 'FAIL'} {w}x{h} -> {tuple(a.shape)}  "
                  f"max|diff|={md:.3e}")
    bad = [r for r in out if not r["identical"]]
    if bad:
        raise RuntimeError(
            f"preprocess_v5.build_transform no longer reproduces "
            f"code_final_fixed_v4.base_preprocess_transform: {bad}. Either torchvision "
            f"changed under one of them or the pin was edited. Until this passes, the "
            f"caches in ablation/caches do not correspond to this pixel path and no "
            f"number resting on them may be reported.")
    if verbose:
        print(f"  pin verified: {describe(img_size, backbone)}")
    return {"cases": out, "rule_id": preprocess_rule_id(img_size, backbone),
            "reference": "code_final_fixed_v4.base_preprocess_transform"}


def assert_tta_views_match_reference(verbose: bool = True) -> Dict[str, Any]:
    """Prove TTA_VIEWS still describes v4's tta_variants -- by AST, without torch.

    TTA_VIEWS is a description of code that lives somewhere else, which is the one kind
    of record that can rot silently: edit tta_variants and every number in
    TTA_WEDGE_AUDIT becomes a measurement of a transform nobody runs any more. So the
    description is checked against the source rather than trusted.

    Read statically, not by calling tta_variants, for two reasons: importing
    code_final_fixed_v4 pulls in torch, and this has to run on the machines where the
    splits and the static checkers run; and the property being checked is the ORDER of
    the view list, which is a fact about the source, not about any tensor.

    Three things are checked, and the third is the one that is easy to forget: the
    slicing rule. `tta_view_names(n)` claims the views are TTA_VIEWS[:max(1, n)]; if v4
    ever changed to `variants[:n]` or reordered its guards, the names this module reports
    for tta=3 would be a different set than the one the extractor averaged.
    """
    import ast
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "code_final_fixed_v4.py")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "tta_variants"), None)
    if fn is None:
        raise RuntimeError(
            "code_final_fixed_v4.tta_variants is gone. TTA_VIEWS and TTA_WEDGE_AUDIT "
            "describe it, so they now describe nothing; re-derive both before any "
            "artifact quotes them.")

    # (1) the names, in source order. Read as the first element of each `(name, fn)`
    # tuple literal rather than as "every string constant in the function": a docstring
    # or a comment-turned-string would otherwise be counted as a view, and the failure
    # message would blame the wrong thing. Sorted by position because ast.walk is
    # breadth-first, so its order is only accidentally source order.
    tuples = sorted((nd for nd in ast.walk(fn) if isinstance(nd, ast.Tuple)),
                    key=lambda nd: (nd.lineno, nd.col_offset))
    names = tuple(nd.elts[0].value for nd in tuples
                  if nd.elts and isinstance(nd.elts[0], ast.Constant)
                  and isinstance(nd.elts[0].value, str))
    declared = tuple(v["name"] for v in TTA_VIEWS)
    if names != declared:
        raise RuntimeError(
            f"TTA_VIEWS no longer matches code_final_fixed_v4.tta_variants: this module "
            f"declares {declared} and the source builds {names}. Whichever changed, "
            f"TTA_WEDGE_AUDIT was measured on the old set and must be re-measured -- and "
            f"the caches in ablation/caches hold the old set, so this is a change of "
            f"condition, not a change of description.")

    # (2) the guard thresholds: view i is admitted at n >= i+1, ascending. Written as a
    # loop rather than a comprehension because the comprehension that does this wants to
    # rebind its own loop variable to `node.test`, which reads like a bug even when it
    # is not.
    guards = []
    for nd in ast.walk(fn):
        if not isinstance(nd, ast.If) or not isinstance(nd.test, ast.Compare):
            continue
        cmp = nd.test
        if (len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.GtE)
                and isinstance(cmp.left, ast.Name)
                and isinstance(cmp.comparators[0], ast.Constant)):
            guards.append(cmp.comparators[0].value)
    guards = tuple(guards)
    if guards != tuple(range(2, len(declared) + 1)):
        raise RuntimeError(
            f"tta_variants' guards are {guards}, expected "
            f"{tuple(range(2, len(declared) + 1))}. tta_view_names() assumes view i is "
            f"added at n >= i+1; it no longer holds.")

    # (3) the slice. Compared structurally rather than as text, because a substring test
    # against source is exactly the check that cannot fail informatively.
    ret = next((nd for nd in ast.walk(fn) if isinstance(nd, ast.Return)), None)
    sl = getattr(ret, "value", None)
    okslice = (isinstance(sl, ast.Subscript) and isinstance(sl.value, ast.Name)
               and isinstance(sl.slice, ast.Slice) and sl.slice.lower is None
               and isinstance(sl.slice.upper, ast.Call)
               and isinstance(sl.slice.upper.func, ast.Name)
               and sl.slice.upper.func.id == "max"
               and len(sl.slice.upper.args) == 2
               and isinstance(sl.slice.upper.args[0], ast.Constant)
               and sl.slice.upper.args[0].value == 1
               and isinstance(sl.slice.upper.args[1], ast.Name))
    if not okslice:
        raise RuntimeError(
            f"tta_variants no longer returns `variants[:max(1, n)]` but "
            f"`{ast.unparse(sl) if sl is not None else None}`. tta_view_names() reports "
            f"the wrong views for every n until it is updated to match.")

    if verbose:
        print(f"  TTA view set verified against code_final_fixed_v4.tta_variants: "
              f"{list(names)}")
    return {"views": list(names), "guards": list(guards),
            "slice": "variants[:max(1, n)]",
            "reference": "code_final_fixed_v4.tta_variants"}


if __name__ == "__main__":
    import sys
    import textwrap

    print(describe())
    for _b in ("resnet18", "densenet121", "vit_b_16"):
        print(f"  {_b:14s} {preprocess_rule_id(224, _b)}")
    print()
    print("\n".join(textwrap.wrap(channel_audit_note(), 96,
                                  initial_indent="  ", subsequent_indent="  ")))
    print()
    print("\n".join(textwrap.wrap(tta_note(3), 96,
                                  initial_indent="  ", subsequent_indent="  ")))
    # No torch needed for this one, so it runs even where the pin check cannot.
    print("\nchecking the TTA view set against code_final_fixed_v4 ...")
    assert_tta_views_match_reference()
    print("\nchecking the pin against code_final_fixed_v4 ...")
    try:
        assert_matches_reference()
    except ImportError as _e:
        print(f"  skipped: {_e} (needs torch; the constants above are still the pin)")
        sys.exit(0)

