"""Explicit real-audio regional attention/generation comparison.

An already imported audio ID and trained local checkpoint are required. All
charts/exports are private test artifacts; library entries, source audio,
settings and learned weights remain untouched. This evaluates actual regional
source choices and density, not instrument accuracy or headset comfort.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import pathlib
import shutil
import sys
import time
import uuid
import zipfile
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import bootstrap  # noqa: F401
from engine.arcaudio import analyze_attention
from engine.arcaudio.settings import attention_source
from engine.generate import generate_map
from engine.music import ensure_music_analysis
from engine.regions import validate_regions
from engine.tempo import ensure_tempo_analysis
from gameplay import add_events
from modding import apply_mods, BeatClock
from projects import _export_project
import projects as projects_module
from quality_benchmark import metrics


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(paths):
    return {str(path.relative_to(ROOT)): checksum(path) for path in paths if path.is_file()}


def event_key(event):
    return (event["time"], event["source"], event.get("reason"), event.get("importance"))


def region_statistics(chart, analysis, regions):
    clock = BeatClock(analysis["bpm"], chart.get("bpmEvents", []))
    rows = []
    for region in regions:
        notes = [note for note in chart["colorNotes"]
                 if region["start"] <= clock.time(note["b"]) < region["end"]]
        sources = collections.Counter(note.get("customData", {}).get("arcforgeSource") for note in notes)
        allowed = {source for source, weight in region.get("sources", {}).items() if weight > 0}
        violations = [source for source in sources
                      if source not in allowed and not (attention_source(source) == "drums" and "drums" in allowed)]
        rows.append({"id": region["id"], "start": region["start"], "end": region["end"],
                     "analysisMode": region.get("analysisMode", "auto"),
                     "notes": len(notes), "actualNps": round(len(notes) / (region["end"] - region["start"]), 4),
                     "sourceCounts": dict(sources),
                     "strictSourceViolations": violations if region.get("analysisMode") == "custom" and region.get("sourcePolicy", "only") == "only" else []})
    return rows


def evaluate(audio_id, quality="balanced", custom_regions=None, audio_folder=None):
    source = (audio_folder or ROOT / "data/audio" / audio_id).resolve()
    analysis_file = source / "analysis.json"
    record = json.loads(analysis_file.read_text("utf-8"))
    record["audioPath"] = str((source / "song.ogg").resolve())
    model = ROOT / "models/style"
    input_paths = [analysis_file, source / "song.ogg"] + list(model.glob("*.npz")) + list(model.glob("*.json"))
    project_paths = list((ROOT / "data/projects").glob("*/project.json"))
    before = fingerprints(input_paths + project_paths)
    folder = ROOT / "data/test-artifacts/mapping-source-regions" / str(time.time_ns())
    folder.mkdir(parents=True)
    isolated_data = folder / "input"
    copied_audio = isolated_data / "audio" / audio_id
    copied_audio.mkdir(parents=True)
    shutil.copy2(source / "song.ogg", copied_audio / "song.ogg")
    record = ensure_tempo_analysis(ensure_music_analysis(record))
    regions = validate_regions(custom_regions if custom_regions is not None else [
        {"id": "full-mix", "name": "Full mix density range", "start": 30., "end": 46.,
         "analysisMode": "all", "npsRange": [8., 13.]},
        {"id": "drums", "name": "Drums only", "start": 60., "end": 76.,
         "analysisMode": "custom", "sources": {"drums": 1.}, "sourcePolicy": "only", "nps": 10.},
        {"id": "spoken", "name": "Spoken only", "start": 254., "end": 260.,
         "analysisMode": "custom", "sources": {"spoken": 1.}, "sourcePolicy": "only"},
    ], record["duration"])
    control_regions = [{k: copy.deepcopy(v) for k, v in region.items() if k not in {"sources", "sourcePolicy"}}
                       | {"analysisMode": "auto"} for region in regions]
    settings = {"bpm": record["bpm"], "nps": 13., "rank": 9, "mode": "Standard", "backend": "local",
                "seed": 197, "style": "Modern", "arcAudioMode": "arcaudio3", "dynamicDensity": True,
                "analysisCompute": {"quality": quality, "device": "auto", "cpuThreads": 0},
                "learnWholeLevel": True, "learnLighting": True, "beatmapVersion": "4",
                "mods": [], "compatibility": "Vanilla", "events": {"lighting": True, "arcs": True},
                "philosophy": {"musicality": 85, "tech": 55, "movement": 55, "variety": 80, "contrast": 90, "stamina": 65}}
    outputs, selections, candidates, rows = {}, {}, {}, []
    for label, directions in [("auto-control", control_regions), ("manual-sources", regions)]:
        started = time.perf_counter()
        analysis = copy.deepcopy(record)
        config = settings | {"mappingRegions": copy.deepcopy(directions)}
        attention = analyze_attention(analysis, mode="arcaudio3", compute=config["analysisCompute"], mapping_regions=directions)
        analysis["_arcAudioAttention"] = attention
        generated = generate_map(analysis, copy.deepcopy(config), model)
        lighting_config = config | {"_arcForgeMapPlan": generated["report"].get("mapPlan", {})}
        chart, mod_info = apply_mods(add_events(generated["beatmap"], analysis, config), analysis, lighting_config)
        destination = folder / label
        destination.mkdir()
        draft = {"id": uuid.uuid4().hex, "name": "Private regional source evaluation", "created": 0, "updated": 0, "revision": 1,
                 "bpm": analysis["bpm"], "environment": "DefaultEnvironment", "audio": record | {"id": audio_id},
                 "config": config, "charts": [{"id": uuid.uuid4().hex, "name": "Expert+", "rank": 9, "mode": "Standard",
                                              "njs": 20., "jumpDistance": 24., "data": chart, "modInfo": mod_info}]}
        with patch.object(projects_module, "DATA", isolated_data):
            archive = _export_project(draft["id"], draft, destination, destination, version="4")
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if name.lower().endswith(".dat"):
                    (destination / name).write_bytes(zipped.read(name))
        measured = metrics(chart, analysis, config)
        mapper_attention = analysis.get("_arcAudioAttention", attention)
        selections[label] = mapper_attention["selectedEvents"]
        candidates[label] = mapper_attention["events"]
        outputs[label] = chart
        row = {"case": label, "seconds": round(time.perf_counter() - started, 3), "metrics": measured,
               "recognition": attention.get("recognition"), "attentionSummary": mapper_attention["summary"],
               "mappingFocus": mapper_attention.get("mappingFocus"), "regions": region_statistics(chart, analysis, directions),
               "regionReport": generated["report"].get("regionReport"), "nativeArchive": str(archive)}
        rows.append(row)
        (destination / "generation-report.json").write_text(json.dumps(generated["report"], indent=2), "utf-8")
        (destination / "selected-events.json").write_text(json.dumps(mapper_attention["selectedEvents"], indent=2), "utf-8")
        print(json.dumps({"case": label, "notes": measured["notes"], "averageNps": measured["averageNps"],
                          "regions": row["regions"], "motion": measured["playability"]}), flush=True)
    focused = [region for region in regions if region.get("analysisMode", "auto") != "auto"]
    def outside(second):
        return not any(region["start"] <= second < region["end"] for region in focused)
    outside_events = {label: [event_key(event) for event in events if outside(event["time"])]
                      for label, events in selections.items()}
    outside_candidates = {label: [event for event in events if outside(event["time"])]
                          for label, events in candidates.items()}
    after = fingerprints(input_paths + project_paths)
    report = {"audioId": audio_id, "durationSeconds": record["duration"], "computeQuality": quality,
              "seed": settings["seed"], "cases": rows,
              "automaticAttentionOutsideRegionsIdentical": outside_events["auto-control"] == outside_events["manual-sources"],
              "automaticCandidateDecisionsOutsideRegionsIdentical": outside_candidates["auto-control"] == outside_candidates["manual-sources"],
              "outsideSelectedEventCounts": {label: len(events) for label, events in outside_events.items()},
              "inputAndExistingProjectFilesUnchanged": before == after, "inputFingerprints": before,
              "limitations": ["The comparison regenerates private charts; it does not mutate a partial authored map.",
                              "Full-chart placement can change after different earlier notes alter hand history. Outside-region preservation in editor continuation is a separate transactional contract.",
                              "Source labels remain detector hypotheses. Selecting Spoken does not transcribe or invent speech.",
                              "A requested NPS range is constrained by observed musical events and motion guards; it is not a density guarantee.",
                              "Native export is generated here; external schema validation and headset review remain separate checks."]}
    (folder / "evaluation.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({"report": str(folder / "evaluation.json"),
                      "outsideAttentionIdentical": report["automaticAttentionOutsideRegionsIdentical"],
                      "inputsUnchanged": report["inputAndExistingProjectFilesUnchanged"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_id")
    parser.add_argument("--quality", choices=["balanced", "deep"], default="balanced")
    parser.add_argument("--regions-json", type=pathlib.Path)
    parser.add_argument("--audio-folder", type=pathlib.Path,
                        help="Explicit existing private test-audio folder; never restores a deleted library entry")
    args = parser.parse_args()
    if pathlib.Path(args.audio_id).name != args.audio_id:
        parser.error("Use an imported audio ID, not a path")
    evaluate(args.audio_id, args.quality,
             json.loads(args.regions_json.read_text("utf-8")) if args.regions_json else None, args.audio_folder)
