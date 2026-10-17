"""Sequential repeated runs; resume replays observations, never completed runs."""
import copy
import json
import math
from pathlib import Path
from uuid import uuid4

from metamodels.base import integer
from pipeline import prepare_runs


def repetition_config(cfg, index):
    integer(cfg.get('repetitions', 1), 'repetitions')
    result = copy.deepcopy(cfg)
    result.pop('repetitions', None)
    offset = index * 100_000
    result['seed'] = int(cfg['seed']) + offset
    design = result['initial_design']
    design['seed'] = int(cfg['initial_design'].get('seed', int(cfg['seed'])+11)) + offset
    if 'seed' in result['metamodel']:
        result['metamodel']['seed'] += offset
    return result


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def observations(path):
    """Only a complete successful prefix is reusable; tolerate a torn final line."""
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines)-1:
                break
            raise
        if row['status'] != 'ok':
            if i != len(lines)-1:
                raise ValueError(f'Nonterminal failed observation in {path}')
            break
        if row['cumulative_calls'] != len(rows)+1 or not math.isfinite(row['y']):
            raise ValueError(f'Invalid observation in {path}')
        rows.append(row)
    return rows


def repeat(cfg, mode, out, resume=False):
    count = integer(cfg.get('repetitions', 1), 'repetitions')
    out = Path(out)
    baseline = repetition_config(cfg, 0)
    baseline.pop('output', None)
    baseline['mode'] = mode
    manifest = out/'experiment.json'
    if out.exists():
        if not resume:
            raise FileExistsError('Output exists; use --resume or a new output directory')
        if not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')) != baseline:
            raise ValueError('Resume configuration differs; only repetitions/output may change')
    # Paired preflight before creating output or calling g.
    first = prepare_runs(repetition_config(cfg, 0), mode)
    out.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        save_json(manifest, baseline)
    summaries = []
    for i in range(count):
        pipelines = first if i == 0 else prepare_runs(repetition_config(cfg, i), mode)
        folder = out/f'rep_{i+1:03d}'
        folder.mkdir(exist_ok=True)
        for name, pipeline in pipelines.items():
            target = folder/name
            completed = target/'summary.json'
            if completed.exists():
                summary = json.loads(completed.read_text(encoding='utf-8'))
                print(f'[skip] {target}', flush=True)
            else:
                saved = folder/f'.{name}_replay.json'
                replay = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else []
                if target.exists():
                    current = observations(target/'g_calls.jsonl')
                    for a, b in zip(replay, current):
                        if any(a[k] != b[k] for k in ('x', 'y', 'iteration', 'role')):
                            raise ValueError(f'Conflicting resume observations: {target}')
                    replay = current if len(current) > len(replay) else replay
                    # Durable replay survives another interruption while rebuilding a prefix.
                    save_json(saved, replay)
                    target.rename(folder/f'.{name}_interrupted_{uuid4().hex[:12]}')
                print(f'[run {i+1}/{count}] {name}; reuse {len(replay)} g calls', flush=True)
                summary = pipeline.run(target, replay=replay)
            summaries.append(dict(repetition=i+1, seed=pipeline.config['seed'],
                                  output=str(target), summary=summary))
        save_json(out/'results.json', summaries)
    return summaries
