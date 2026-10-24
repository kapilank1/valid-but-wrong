"""Arms race round 2: the ppo_r2 attack, its round-1 hardened shadow, and loading a specific saved L3."""

from __future__ import annotations

import json

import numpy as np
import torch

from vbw.attacker import registry
from vbw.attacker.env import MODELS, PpoR2Attack
from vbw.attacker.registry import list_attacks, make_attack
from vbw.detector.features import L3_CONTEXT, feature_names
from vbw.detector.l3 import HardenedDetector, L3Model
from vbw.detector.nets import WindowClassifier, export_onnx
from vbw.schema import FALCON9, Frame


def test_registry_lists_ppo_r2_for_falcon9_with_availability_from_its_model_file():
    entry = next(a for a in list_attacks() if a["name"] == "ppo_r2")
    assert entry["schemas"] == ["falcon9"]
    assert entry["available"] == registry._model_path("falcon9", "ppo_r2").exists()
    assert registry._model_path("falcon9", "ppo_r2").name == "ppo_falcon9_r2.zip"
    assert registry._model_path("falcon9").name == "ppo_falcon9.zip"


def test_ppo_r2_reports_its_own_name_and_is_identity_without_a_model(tmp_path):
    atk = make_attack("ppo_r2", FALCON9, start_t=0.0, model_path=str(tmp_path / "missing.zip"))
    assert isinstance(atk, PpoR2Attack) and atk.name == "ppo_r2" and atk.loaded is False
    assert atk.detector_name == "hardened"
    f = Frame("f9", FALCON9.schema_id, 0, 1.0, {c: 1.0 for c in FALCON9.channels})
    assert atk.apply(f).channels == f.channels


def test_ppo_r2_defaults_to_the_r2_policy_and_the_round_1_shadow(tmp_path):
    atk = PpoR2Attack(schema=FALCON9, model_path=str(tmp_path / "missing.zip"))
    r1 = MODELS / "l3_falcon9_r1.onnx"
    if r1.exists():
        assert atk.shadow_l3 == str(r1)
        assert isinstance(atk._shadow, HardenedDetector)
        assert atk._shadow.l3.meta == json.loads((MODELS / "l3_falcon9_r1_meta.json").read_text())
    assert PpoR2Attack(schema=FALCON9, shadow_l3="x").model_path == str(MODELS / "ppo_falcon9_r2.zip")


def test_l3_model_from_file_reads_the_sibling_meta(tmp_path):
    torch.manual_seed(0)
    clf = WindowClassifier(len(feature_names(FALCON9)), hidden=4)
    path = tmp_path / "l3_falcon9_r9.onnx"
    export_onnx(clf, np.zeros((2, L3_CONTEXT, len(feature_names(FALCON9))), np.float32), path, "logit")
    (tmp_path / "l3_falcon9_r9_meta.json").write_text(json.dumps({"logit_threshold": 3.5, "context": L3_CONTEXT}))
    model = L3Model.from_file(path)
    assert model.logit_threshold == 3.5 and model.context == L3_CONTEXT
    assert model.logit(np.zeros((1, L3_CONTEXT, len(feature_names(FALCON9))), np.float32)).shape == (1,)
