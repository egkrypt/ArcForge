"""Explicit local full-song compute comparison, never an accuracy benchmark.

Requires an already imported audio ID and installed local recognition weights.
Writes private diagnostics under data/test-artifacts only. No network, training,
authored chart mutation, or attention-setting persistence is performed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import os
import pathlib
import sys
import time
import faulthandler
import types
from unittest.mock import patch

faulthandler.enable()
faulthandler.dump_traceback_later(90., repeat=True)
print("Starting local analysis benchmark imports", flush=True)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401
import numpy as np
from engine.features import decode_audio
from engine import recognition
from engine.arcaudio import analyze_evidence
from engine.arcaudio.phrases import build_phrases


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_measurements(result):
    return {key: value for key, value in result.items()
            if key not in {"scores", "times", "embeddings"}}


def process_memory():
    """RSS/peak resident bytes without adding a benchmark-only dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("pageFaults", wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ["peakWorkingSet", "workingSet", "peakPagedPool", "pagedPool",
                                                 "peakNonPagedPool", "nonPagedPool", "pagefile", "peakPagefile", "private"]]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        read_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        read_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if read_memory(current_process(), ctypes.byref(counter), counter.cb):
            return {"processRSSBytes": int(counter.workingSet), "peakProcessRSSBytes": int(counter.peakWorkingSet)}
    try:
        import psutil
        return {"processRSSBytes": psutil.Process().memory_info().rss, "peakProcessRSSBytes": None}
    except ImportError:
        return {"processRSSBytes": None, "peakProcessRSSBytes": None}


def attention(analysis, result):
    evidence = recognition.evidence_from_scores(analysis, result["scores"], result["times"])
    phrases = build_phrases(analysis, evidence["frames"], evidence["events"])
    selected = analyze_evidence(evidence["frames"], evidence["events"], phrases,
                                beat_times=analysis.get("beatTimes", []))
    def summary(start, end):
        frames = [frame for frame in selected["frames"] if start <= frame["time"] < end]
        events = [event for event in selected["selectedEvents"] if start <= event["time"] < end]
        def mean(values):
            return round(float(np.mean(values)), 6) if values else None
        return {"start": start, "end": end, "frames": len(frames), "selectedEvents": len(events),
                "primaryCounts": dict(collections.Counter(f["primarySource"] for f in frames)),
                "meanInstrumentPosteriors": {
                    source: mean([f.get("instrumentEvidence", {}).get(source, 0.) for f in frames])
                    for source in ["piano", "guitar", "synthesizer", "plucked_strings", "bowed_strings"]},
                "meanSourceStrengths": {source: mean([f["sources"][source] for f in frames])
                                        for source in recognition.SOURCES}}
    return {"summary": selected["summary"],
            "selectedEventHash": hashlib.sha256(json.dumps([(e["time"], e["source"]) for e in selected["selectedEvents"]]).encode()).hexdigest(),
            "windows": [summary(0., 15.), summary(20., 45.), summary(254., 260.)]}, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_id")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--quality", choices=["balanced", "deep"], default="balanced")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--compare", help="Earlier phase name in the same artifact folder")
    parser.add_argument("--baseline-snapshot", type=pathlib.Path,
                        help="Private diagnostic folder containing recognition-before-compute.py and yamnet-before-compute.py")
    parser.add_argument("--audio-folder", type=pathlib.Path,
                        help="Explicit existing private test-audio folder; never restores a deleted library entry")
    parser.add_argument("--pipeline", action="store_true",
                        help="Measure actual source-recognition orchestration, including Deep CPU/GPU overlap, in a private cache")
    args = parser.parse_args()
    if pathlib.Path(args.audio_id).name != args.audio_id:
        raise ValueError("Use an imported audio ID, not a path")
    folder = ROOT / "data/test-artifacts/analysis-compute" / args.audio_id
    folder.mkdir(parents=True, exist_ok=True)
    audio_folder = args.audio_folder or ROOT / "data/audio" / args.audio_id
    analysis_path = audio_folder / "analysis.json"
    analysis = json.loads(analysis_path.read_text("utf-8"))
    audio_path = analysis_path.parent / "song.ogg"
    analysis["audioPath"] = str(audio_path.resolve())
    global recognition
    if args.baseline_snapshot:
        name = "engine._benchmark_baseline_recognition"
        yamnet = types.ModuleType(name + ".yamnet")
        yamnet.__file__ = str(ROOT / "engine/recognition/yamnet.py")
        yamnet.__package__ = name
        sys.modules[yamnet.__name__] = yamnet
        exec(compile((args.baseline_snapshot / "yamnet-before-compute.py").read_text("utf-8"), yamnet.__file__, "exec"), yamnet.__dict__)
        previous = types.ModuleType(name)
        previous.__file__ = str(ROOT / "engine/recognition/__init__.py")
        previous.__package__ = name
        previous.__path__ = [str(ROOT / "engine/recognition")]
        sys.modules[name] = previous
        exec(compile((args.baseline_snapshot / "recognition-before-compute.py").read_text("utf-8"), previous.__file__, "exec"), previous.__dict__)
        recognition = previous
    input_hashes = {"analysis": sha256(analysis_path), "audio": sha256(audio_path),
                    "recognitionWeights": sha256(recognition.WEIGHTS)}
    source_hashes = {str(path.relative_to(ROOT)): sha256(path)
                     for path in [ROOT / "engine/recognition/__init__.py", ROOT / "engine/recognition/yamnet.py"]}
    if args.baseline_snapshot:
        source_hashes = {path.name: sha256(path) for path in [args.baseline_snapshot / "recognition-before-compute.py",
                                                            args.baseline_snapshot / "yamnet-before-compute.py"]}
    print(json.dumps({"phase": args.phase, "stage": "loading", "sourceHashes": source_hashes}), flush=True)
    hardware_start = time.perf_counter()
    torch = recognition._torch()
    cuda = bool(torch.cuda.is_available())
    hardware = {"logicalCPUs": os.cpu_count(), "torchThreadsBefore": torch.get_num_threads(),
                "cudaAvailable": cuda,
                "gpuName": torch.cuda.get_device_name(0) if cuda else None,
                "gpuTotalBytes": torch.cuda.get_device_properties(0).total_memory if cuda else None}
    try:
        import psutil
        hardware["physicalCPUs"] = psutil.cpu_count(logical=False)
        hardware["totalRAMBytes"] = psutil.virtual_memory().total
    except ImportError:
        psutil = None
    hardware["probeSeconds"] = round(time.perf_counter() - hardware_start, 5)
    print(json.dumps({"phase": args.phase, "stage": "hardware", "hardware": hardware}), flush=True)
    decode_start = time.perf_counter()
    samples = decode_audio(audio_path, sample_rate=16000)
    decode_seconds = time.perf_counter() - decode_start
    kwargs = {"device": args.device}
    if "settings" in inspect.signature(recognition.classify_samples).parameters:
        kwargs["settings"] = {"quality": args.quality, "device": args.device, "cpuThreads": args.threads}
    elif args.quality != "balanced" or args.threads:
        raise RuntimeError("This source version does not implement requested compute settings")
    def run_once():
        if not args.pipeline:
            return recognition.classify_samples(samples, **kwargs)
        config = {"quality": args.quality, "device": args.device, "cpuThreads": args.threads}
        with patch.object(recognition, "ROOT", folder / "private-runtime"):
            evidence = recognition.recognize(analysis, settings=config, force=True)
        caches = list((folder / "private-runtime/data/recognition-cache").glob("*.npz"))
        if len(caches) != 1:
            raise AssertionError("The private diagnostic cache must contain exactly one source-recognition result")
        with np.load(caches[0], allow_pickle=False) as saved:
            scores, times = saved["scores"].copy(), saved["times"].copy()
        return {**evidence["performance"], "scores": scores, "times": times}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    cpu_start = os.times()
    wall_start = time.perf_counter()
    result = run_once()
    if cuda:
        torch.cuda.synchronize()
    inference_wall = time.perf_counter() - wall_start
    cpu_end = os.times()
    resources = {"processCPUSeconds": round(cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system, 5),
                 "meanCPUCoreEquivalents": round((cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system) / inference_wall, 5),
                 "torchThreadsAfter": torch.get_num_threads(),
                 "peakCUDATensorBytes": torch.cuda.max_memory_allocated() if cuda else None,
                 "peakCUDAReservedBytes": torch.cuda.max_memory_reserved() if cuda else None,
                 **process_memory()}
    np.savez_compressed(folder / f"{args.phase}-scores.npz", scores=result["scores"], times=result["times"])
    print(json.dumps({"phase": args.phase, "stage": "inference", "measurements": public_measurements(result),
                      "wallSeconds": round(inference_wall, 5), "resources": resources}), flush=True)
    repeat = None
    if args.repeat:
        repeated = run_once()
        repeat = {"measurements": public_measurements(repeated),
                  "timesExactlyEqual": bool(np.array_equal(result["times"], repeated["times"])),
                  "maxAbsoluteScoreDifference": float(np.max(np.abs(result["scores"] - repeated["scores"])))}
    attention_start = time.perf_counter()
    attention_summary, selected = attention(analysis, result)
    attention_seconds = time.perf_counter() - attention_start
    comparison = None
    if args.compare:
        with np.load(folder / f"{args.compare}-scores.npz", allow_pickle=False) as old:
            old_scores, old_times = old["scores"], old["times"]
        old_summary, old_selected = attention(analysis, {"scores": old_scores, "times": old_times})
        frame_times = np.asarray([f["time"] for f in selected["frames"]])
        old_interp = np.stack([np.interp(frame_times, old_times, old_scores[:, c]) for c in range(521)], 1)
        new_interp = np.stack([np.interp(frame_times, result["times"], result["scores"][:, c]) for c in range(521)], 1)
        old_keys = {(e["time"], e["source"]) for e in old_selected["selectedEvents"]}
        new_keys = {(e["time"], e["source"]) for e in selected["selectedEvents"]}
        comparison = {"baselinePhase": args.compare, "baselinePatchCount": len(old_times),
                      "currentPatchCount": len(result["times"]),
                      "interpolatedClassPosteriorMeanAbsoluteDifference": float(np.mean(np.abs(new_interp - old_interp))),
                      "interpolatedClassPosteriorMaxAbsoluteDifference": float(np.max(np.abs(new_interp - old_interp))),
                      "primaryFramesChanged": sum(a["primarySource"] != b["primarySource"] for a, b in zip(selected["frames"], old_selected["frames"])),
                      "totalPrimaryFrames": len(selected["frames"]),
                      "selectedEventIntersection": len(old_keys & new_keys),
                      "selectedEventUnion": len(old_keys | new_keys),
                      "baselineAttention": old_summary}
    report = {"audioID": args.audio_id, "durationSeconds": len(samples) / 16000,
              "phase": args.phase, "requested": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}, "inputHashes": input_hashes,
              "sourceHashes": source_hashes, "hardware": hardware,
              "operation": "full_recognition" if args.pipeline else "classifier_only",
              "decodeSeconds": round(decode_seconds, 5), "inferenceWallSeconds": round(inference_wall, 5),
              "inference": public_measurements(result), "resources": resources,
              "attentionSeconds": round(attention_seconds, 5), "attention": attention_summary,
              "repeat": repeat, "comparison": comparison,
              "inputFilesUnchanged": input_hashes == {"analysis": sha256(analysis_path), "audio": sha256(audio_path),
                                                       "recognitionWeights": sha256(recognition.WEIGHTS)},
              "limitations": ["This is a runtime, coverage, and reproducibility comparison on one machine and one song.",
                              "No reference instrument labels are available; no detection-accuracy or musical-quality improvement is claimed.",
                              "More overlapping classifier windows do not change the 0.96-second receptive field or create precise note timings.",
                              "CPU core equivalents measure this process CPU time, not whole-device utilization; CUDA counters exclude driver/context and other processes."]}
    (folder / f"{args.phase}.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({"phase": args.phase, "stage": "complete", "report": str(folder / f"{args.phase}.json"),
                      "unchanged": report["inputFilesUnchanged"], "attention": attention_summary["summary"],
                      "repeat": repeat, "comparison": {k: v for k, v in (comparison or {}).items() if k != "baselineAttention"}}), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
