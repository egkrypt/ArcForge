"""Explicit local benchmark; does not retrain or modify authored projects."""
import json, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import bootstrap
from engine.dataset import scan_dataset, import_references_fast
from engine.trainer import FEATURE_SCHEMA_VERSION
from engine.composition import FEATURE_SCHEMA_VERSION as COMPOSITION_SCHEMA

started = time.perf_counter(); cold = scan_dataset(ROOT / 'data/dataset'); cold_seconds = time.perf_counter() - started
created = sum(archive.get('metadataCache') == 'created' for archive in cold['archives'])
print(json.dumps({'stage': 'firstProcessScan', 'seconds': cold_seconds, 'archives': cold['archiveCount'], 'metadataCreated': created, 'errors': len(cold['errors'])}), flush=True)
started = time.perf_counter(); warm = scan_dataset(ROOT / 'data/dataset'); warm_seconds = time.perf_counter() - started
print(json.dumps({'stage': 'warmScan', 'seconds': warm_seconds, 'archives': warm['archiveCount'], 'errors': len(warm['errors'])}), flush=True)
first = import_references_fast(ROOT.parent / 'Learn from these levels', ROOT / 'data/dataset')
print(json.dumps({'stage': 'firstIncrementalImport', 'seconds': first['seconds'], 'processed': first['processed'], 'unchanged': first['unchanged'], 'rejected': len(first['rejected'])}), flush=True)
second = import_references_fast(ROOT.parent / 'Learn from these levels', ROOT / 'data/dataset')
report = {'datasetArchives': warm['archiveCount'], 'datasetCharts': warm['mapCount'], 'noteCount': warm['noteCount'],
          'firstProcessScanSeconds': round(cold_seconds, 4), 'warmScanSeconds': round(warm_seconds, 4),
          'firstScanMetadataCreated': created, 'firstScanMetadataReused': cold['archiveCount'] - created,
          'coldScanMeasured': created == cold['archiveCount'],
          'firstImport': {k: first[k] for k in ['seconds', 'processed', 'unchanged', 'sourceCount']},
          'warmImport': {k: second[k] for k in ['seconds', 'processed', 'unchanged', 'sourceCount']},
          'errors': warm['errors'], 'rejectedImports': second['rejected'],
          'placementExistingSchemaCaches': len(list((ROOT / 'models/style/feature-cache').glob('*-' + FEATURE_SCHEMA_VERSION + '.npz'))),
          'compositionExistingSchemaCaches': len(list((ROOT / 'models/style/composition-cache').glob('*-' + COMPOSITION_SCHEMA + '.npz')))}
(ROOT / 'data/import13-benchmark.json').write_text(json.dumps(report, indent=2), 'utf-8')
print(json.dumps({key: value for key, value in report.items() if key not in ['errors', 'rejectedImports']} |
                 {'errors': len(report['errors']), 'rejectedImports': len(report['rejectedImports']),
                  'rejectionSamples': report['rejectedImports'][:3]}, indent=2), flush=True)
