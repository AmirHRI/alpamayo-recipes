"""Audit local distillation artifacts without loading model weights."""

from __future__ import annotations

import json
import math
import ast
import csv
import copy
import hashlib
import re
from pathlib import Path
from statistics import mean

import yaml


RECIPE = Path(__file__).resolve().parents[1]
ROOT = Path('/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training')
ISSUES = []


def load_yaml(path):
    with path.open() as stream:
        return yaml.safe_load(stream) or {}


def load_json(path):
    with path.open() as stream:
        return json.load(stream)


def metric_rows(payload):
    if isinstance(payload, dict):
        payload = payload.get('per_clip', payload)
        if isinstance(payload, dict):
            payload = list(payload.values())
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)
            and 'clip_id' in row and 'ade' in row and 'min_ade' in row]


def summarize_metrics(path):
    rows = metric_rows(load_json(path))
    if not rows:
        return None
    identities = [row['clip_id'] for row in rows]
    valid = [row for row in rows if all(
        isinstance(row[key], (int, float)) and math.isfinite(row[key])
        for key in ('ade', 'min_ade'))]
    return {
        'result_path': str(path), 'n': len(rows),
        'unique_clips': len(set(identities)), 'invalid_rows': len(rows) - len(valid),
        'clip_set_sha256': hashlib.sha256('\n'.join(sorted(identities)).encode()).hexdigest(),
        'ade': mean(row['ade'] for row in valid) if valid else None,
        'min_ade': mean(row['min_ade'] for row in valid) if valid else None,
    }


def training_runs(history):
    records = []
    completed = completed_training_logs()
    for config_path in sorted(ROOT.glob('*/config.yaml')):
        config = load_yaml(config_path)
        states = []
        for path in config_path.parent.glob('checkpoint-*/trainer_state.json'):
            state = load_json(path)
            states.append((float(state.get('epoch') or 0), state.get('global_step', 0), path))
        latest = max(states, default=(0, 0, None))
        source = config_path
        notes = []
        proven = [item for item in completed
                  if item[1].get('paths', {}).get('output_dir') == str(config_path.parent)
                  and abs(item[2] - latest[0]) < 0.001]
        if proven:
            source, config, _ = proven[-1]
            notes.append('Training settings recovered from completed training log')
        if latest[1] > 0 and 0 < config.get('trainer', {}).get('max_steps', -1) < latest[1]:
            candidates = [(path, saved) for path, saved in history
                          if saved.get('paths', {}).get('output_dir') == str(config_path.parent)
                          and saved.get('trainer', {}).get('max_steps', -1) <= 0
                          and not saved.get('evaluate', {}).get('per_clip_output')
                          and saved.get('trainer', {}).get('num_train_epochs') == latest[0]]
            if candidates:
                source, config = candidates[-1]
                notes.append('Root config overwritten by shorter run; historical config used')
            else:
                notes.append('Root config conflicts with checkpoint; training config unresolved')
        records.append({
            'run_dir': str(config_path.parent), 'epoch': latest[0],
            'global_step': latest[1], 'state_path': str(latest[2] or ''),
            'config': config, 'config_source': str(source), 'notes': '; '.join(notes),
            'run_id': config_path.parent.name,
        })
    legacy = next((item for item in completed if item[0].name == 'kavatrain_365.out'), None)
    if legacy:
        path, config, epoch = legacy
        records.append({'run_dir': config['paths']['output_dir'], 'epoch': epoch,
                        'global_step': 7191, 'state_path': str(path), 'config': config,
                        'config_source': str(path), 'run_id': 'kava_historical_3epoch_7191',
                        'notes': 'Historical 3-epoch run; directory reused; checkpoint-7191 no longer present'})
    return records


def completed_training_logs():
    completed = []
    for path in sorted(ROOT.glob('*.out')):
        text = path.read_text(errors='replace')
        finishes = re.findall(r"\{'train_runtime':[^\n]+", text)
        match = re.search(r'\] Configs:\s*\n(\{\n.*?\n\})', text, re.S)
        if not finishes or not match:
            continue
        try:
            config = ast.literal_eval(match[1].replace(chr(0x2502), ' '))
            finish = ast.literal_eval(finishes[-1])
        except (SyntaxError, ValueError):
            ISSUES.append(f'Could not parse completed training log: {path}')
            continue
        completed.append((path, config, float(finish.get('epoch', 0))))
    return sorted(completed, key=lambda item: item[0].stat().st_mtime)


def configuration_history():
    history = []
    for path in sorted(RECIPE.parent.glob('*/outputs/*/*/.hydra/config.yaml')):
        try:
            history.append((path, load_yaml(path)))
        except yaml.YAMLError:
            ISSUES.append(f'Malformed historical config: {path}')
    return history


def evaluation_configs(history):
    indexed = {}
    for path, config in history:
        output = config.get('evaluate', {}).get('per_clip_output')
        if output and isinstance(output, str):
            indexed.setdefault(output, []).append((path, config))
        sweep = config.get('sweep', {})
        if sweep.get('tag'):
            folders = [Path(sweep['out_dir'])] if sweep.get('out_dir') else [ROOT / 'stepsweep', ROOT / 'camsweep']
            for folder in folders:
                for result in folder.glob(f'{sweep["tag"]}_*.json'):
                    if sweep.get('only') and result.stem != f'{sweep["tag"]}_{sweep["only"]}':
                        continue
                    indexed.setdefault(str(result), []).append((path, config))
    return indexed


def log_config(path):
    config = {'data': {}, 'evaluate': {}, 'model': {}}
    if not path.exists():
        return config
    text = path.read_text(errors='replace')
    for heading, section in [('Dataset Configs', 'data'), ('Evaluate Configs', 'evaluate')]:
        match = re.search(heading + r':\s*\n(\{\n.*?\n\})', text, re.S)
        if match:
            cleaned = re.sub(r'\x1b\[[0-9;]*m', '', match[1]).replace(chr(0x2502), ' ')
            try:
                value = ast.literal_eval(cleaned)
                config[section] = {'val_dataset': value} if section == 'data' else value
            except (SyntaxError, ValueError):
                ISSUES.append(f'Could not parse {heading}: {path}')
    matches = re.findall(r'/data/[^\s\'\";,<>]+/checkpoint-\d+', text)
    if matches and not config['evaluate'].get('eval_ckpt'):
        config['model']['checkpoint_path'] = matches[0]
    return config


def dataset_fields(dataset):
    preprocess = dataset.get('vla_preprocess_args', {})
    annotations = dataset.get('annotations_path') or ''
    return {
        'dataset_class': dataset.get('_target_', ''),
        'dataset_root': dataset.get('local_dir', ''),
        'clip_filter': dataset.get('clip_uuid_filter', ''),
        'annotations': annotations,
        'chunks': dataset.get('chunk_ids', ''),
        'cameras': json.dumps(dataset.get('cameras', [0, 1, 2, 3])) if dataset else 'unknown',
        'timing': 'annotation anchored' if annotations else (
            'default keyframe' if dataset.get('use_default_keyframe') else 'unknown'),
        'route_in_prompt': 'route' in preprocess.get('components_order', []),
        'camera_ids': preprocess.get('include_camera_ids', 'unknown'),
        'frame_numbers': preprocess.get('include_frame_nums', 'unknown'),
    }


def checkpoint_from(config):
    model = config.get('model', {})
    return (config.get('evaluate', {}).get('eval_ckpt')
            or model.get('kava_checkpoint_path') or model.get('checkpoint_path')
            or model.get('eos_checkpoint_path') or '')


def objective(config):
    model = config.get('model', {})
    details = {key: model[key] for key in ('kd', 'kava', 'cd', 'latent_loss_weight',
               'latent_cosine_weight', 'cotrain_vlm', 'stop_grad_from_vlm') if key in model}
    return json.dumps(details, sort_keys=True)


def family(config):
    model = config.get('model', {})
    if 'cd' in model:
        return 'Consistency / endpoint distillation'
    if 'kava' in model:
        return 'KAVA latent-slot distillation and controls'
    if 'kd' in model:
        return 'VLM cache / block / field distillation and CE controls'
    return 'Expert-on-student / supervised controls'


def evaluated_record(summary, configs, run_index):
    path = Path(summary['result_path'])
    candidates = configs.get(str(path), [])
    config_path, config = candidates[-1] if candidates else ('', {})
    log_path = path.with_suffix('.log')
    fallback = {
        'b2c2cam_step2000_2cam': ('camsweep/b2c2cam_step2500_2cam.json', 'eval_b2c_2000.log'),
        'eos4b_3cam': ('camsweep/eos4b_2cam.json', 'sw_eos4b_3cam.log'),
        'eos4b_4cam': ('camsweep/eos4b_2cam.json', 'sw_eos4b_4cam.log'),
        'nav_cdp28_e2_s3126_k1': ('stepsweep/nav_cdp28_e1_s1600_k1.json', 'cdeval_565.out'),
        'nav_cdp28_e2_s3126_k2': ('stepsweep/nav_cdp28_e1_s1600_k2.json', 'cdeval_565.out'),
    }
    inferred = not candidates and path.stem in fallback
    if inferred:
        donor, log_name = fallback[path.stem]
        config_path, config = configs[str(ROOT / donor)][-1]
        config = copy.deepcopy(config)
        log_path = ROOT / log_name
        verified_checkpoint = checkpoint_from(log_config(log_path))
        if path.stem.startswith('nav_cdp28'):
            config['model']['teacher_checkpoint_path'] = verified_checkpoint
        else:
            config['model']['checkpoint_path'] = verified_checkpoint
            if path.stem.startswith('eos4b'):
                config['model']['teacher_checkpoint_path'] = verified_checkpoint
    log_aliases = {'kava_perclip_slots': 'kavaeval_slots',
                   'kava_perclip_zeroed': 'kavaeval_zeroed',
                   'kava_perclip_zeroed2': 'kavaeval_zeroed2'}
    if path.stem in log_aliases:
        log_path = ROOT / (log_aliases[path.stem] + '.log')
    logged = log_config(log_path)
    config = {**config, 'data': {**config.get('data', {}), **logged.get('data', {})}}
    config['evaluate'] = {**config.get('evaluate', {}), **logged.get('evaluate', {})}
    checkpoint = checkpoint_from(config) or checkpoint_from(logged)
    notes = []
    if inferred:
        notes.append('Protocol inferred from sibling config; checkpoint verified in launch log')
    if len(candidates) > 1:
        notes.append(f'{len(candidates)} historical configs reference result; latest plus log used')
    if not config_path:
        notes.append('No matching Hydra config; log metadata used' if log_path.exists()
                     else 'No matching Hydra config or sidecar log')
    model = config.get('model', {})
    expert_source = model.get('teacher_checkpoint_path', '')
    if isinstance(expert_source, str) and '/output_cd_' in expert_source:
        checkpoint = expert_source
    if isinstance(checkpoint, str) and '${' in checkpoint:
        checkpoint = checkpoint_from(logged)
    match = re.search(r'(.+)/checkpoint-(\d+)$', str(checkpoint))
    run_dir = match[1] if match else ''
    step = int(match[2]) if match else 0
    run = run_index.get(run_dir)
    if run_dir.endswith('output_stage1_kava_cosmos2b_lcdrive') and step == 7191:
        run = run_index.get('kava_historical_3epoch_7191')
    epoch = ''
    state_path = Path(checkpoint) / 'trainer_state.json' if checkpoint else None
    if state_path and state_path.exists():
        epoch = float(load_json(state_path).get('epoch') or 0)
    elif run and run['global_step'] and step:
        epoch = round(step * run['epoch'] / run['global_step'], 4)
        notes.append('Checkpoint absent; epoch estimated from retained step/epoch ratio')
    dataset = config.get('data', {}).get('val_dataset', {})
    fields = dataset_fields(dataset)
    sweep = config.get('sweep', {})
    if path.parent.name == 'stepsweep' and config:
        fields['cameras'] = json.dumps(sweep.get('cameras', [1, 3]))
    if path.parent.name == 'camsweep':
        camera_sets = {'1cam': [1], '2cam': [1, 3], '3cam': [0, 1, 2], '4cam': [0, 1, 2, 3]}
        fields['cameras'] = json.dumps(camera_sets.get(path.stem.split('_')[-1], []))
    nfe_match = re.search(r'(?:_nfe|_k)(\d+)', path.stem)
    metrics = config.get('evaluate', {}).get('metric_runner', {}).get('metrics', [])
    nfe = (int(nfe_match[1]) if nfe_match else
           next((item.get('diffusion_kwargs', {}).get('inference_step') for item in metrics
                 if item.get('diffusion_kwargs', {}).get('inference_step') is not None), ''))
    token = path.stem.startswith(('evalhf_', 'e3_', 'eT2_', 'ectl_', 'kava_perclip'))
    head = 'token' if token else 'action expert'
    if not token and not nfe:
        nfe = '10 (default; verify)' if config else 'unknown'
    status = 'included'
    if 'smoke' in str(path).lower() or summary['n'] < 100:
        status = 'excluded: smoke / tiny diagnostic eval'
    elif isinstance(epoch, (int, float)) and epoch < 0.999:
        status = 'excluded: checkpoint below one epoch'
    elif run and run['epoch'] < 0.999:
        status = 'excluded: training below one epoch'
    elif not run:
        status = 'teacher / baseline reference' if 'teacher' in path.stem or path.stem.startswith('evalhf_10b') else 'reference / unresolved training match'
    if path.stem.startswith('structinit_'):
        status = 'post-training weight-edit diagnostic'
    if summary['invalid_rows'] or summary['unique_clips'] != summary['n']:
        notes.append('WARNING: nonfinite metrics or duplicate clip IDs')
    return {**summary, 'status': status, 'run_dir': run_dir, 'run_id': run['run_id'] if run else '', 'checkpoint': checkpoint,
            'checkpoint_exists': Path(checkpoint).is_dir() if checkpoint else False,
            'epoch': epoch, 'head': head, 'nfe': nfe, **fields,
            'expert_source': expert_source,
            'eval_model_config': json.dumps(model, sort_keys=True),
            'model_target': model.get('_target_', ''),
            'config_source': str(config_path), 'log_source': str(log_path) if log_path.exists() else '',
            'notes': '; '.join(notes)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def loss_label(config):
    model = config.get('model', {})
    if 'kd' in model:
        kd = model['kd']
        terms = []
        for key, label, default in [('ce_weight', 'CE', 1), ('kd_weight', 'logit KD', 0),
                                     ('kv_weight', 'KV', 0), ('block_weight', 'block', 0),
                                     ('field_weight', 'field', 0)]:
            weight = kd.get(key, default)
            if weight:
                terms.append(f'{weight:g}*{label}')
        details = []
        if kd.get('block_weight'):
            details.append('t=' + kd.get('block_timestep', 'zero'))
            details.append('span=' + str(kd.get('block_span', 1)))
        for key in ('block_span_mix', 'block_span_mix_weight', 'block_layer_weights', 'kv_layer_bands'):
            if key in kd:
                details.append(f'{key}={kd[key]}')
        if kd.get('layer_mix'):
            details.append('layer mixing')
        return ' + '.join(terms) + ('; ' + ', '.join(details) if details else '')
    if 'kava' in model:
        kava = model['kava']
        return (f'CE + {model.get("latent_loss_weight", 0):g}*latent + '
                f'{kava.get("kv_loss_weight", 1):g}*KV; M={kava.get("num_slots")}, '
                f'T={kava.get("jacobi_iters")}, {kava.get("kv_loss_type")}')
    if 'cd' in model:
        cd = model['cd']
        gt = cd.get('x0_gt_weight', 0)
        teacher = cd.get('x0_teacher_weight', 0)
        if cd.get('x0_source') == 'teacher' and not teacher:
            teacher, gt = gt, 0
        return (f'{cd.get("cd_weight", 1):g}*CD + {teacher:g}*teacher endpoint + {gt:g}*GT endpoint; '
                f'teacher={cd.get("teacher_source", "online/self")}'
                + (f'; terminal fraction={cd["terminal_rung_frac"]}' if 'terminal_rung_frac' in cd else ''))
    return 'Supervised action flow matching; VLM frozen (EoS control)'


DATASET_KEYS = ['dataset_class', 'dataset_root', 'clip_filter', 'annotations', 'chunks',
                'cameras', 'timing', 'route_in_prompt', 'camera_ids', 'frame_numbers']


def dataset_key(row):
    return tuple(str(row[key]) for key in DATASET_KEYS)


def render_report(destination, training, evaluations):
    eligible = [row for row in training if row['status'] == 'included']
    kept = [row for row in evaluations if row['status'] in ('included', 'teacher / baseline reference')]
    registry = {}
    for row in eligible + kept:
        registry.setdefault(dataset_key(row), f'D{len(registry) + 1:02}')
    ids = {row['run_id']: f'T{number:02}' for number, row in enumerate(eligible, 1)}
    lines = [
        '# Distillation Ablation Inventory', '',
        'Local artifact audit, 2026-09-17. Metrics recomputed from saved per-clip JSON, not copied from documentation.', '',
        f'**{len(eligible)} training records in {len({row["run_dir"] for row in eligible})} directories; '
        f'{sum(row["status"] == "included" for row in evaluations)} non-smoke trained-model evaluations; '
        f'{sum(row["status"] == "teacher / baseline reference" for row in evaluations)} teacher/baseline evaluations.**', '',
        '## Scope And Rules', '',
        f'- Training and result paths below are relative to `{ROOT}` (the artifact root).',
        '- Scope: the distillation recipe artifact root, its top-level and camera/step-sweep per-clip results, and historical Hydra configs across neighboring recipes. No remote W&B/Hugging Face history was fetched.',
        '- Keep training with observed epoch >= 1, including interrupted longer runs and resumed stages. Configured epochs alone do not qualify a run. Epoch values on resumes are cumulative, not additional epochs.',
        '- Explicit tiny-data smoke jobs are excluded even when their counters exceed one epoch. Completed 2-30-step jobs were identified this way; see completed_log_audit.csv for the full list.',
        '- Exclude evaluated checkpoints below one epoch, explicit smoke evaluations, and tiny diagnostic sets (<100 clips). Keep 500-clip KAVA evaluations and the 1,000-clip validation studies.',
        '- A retained checkpoint supplies the epoch. If deleted, estimate it from a retained step/epoch ratio and flag the CSV row. This estimate assumes unchanged steps per epoch.',
        '- ADE and minADE are metres, lower is better. Different heads, cameras, expert depth, annotation timing, prompt flags, or NFE are separate protocols, not clean loss-only comparisons.',
        '- NFE is the number of denoising steps. Token-head rows have no NFE. Expert rows without an explicit override use the documented 10-step default, marked in the CSV.',
        '- Navigation annotations use GT future direction (`--horizon-start 0` in the recorded study). These are annotation-conditioned, potentially label-leaking evaluations, not leak-free planner-route results. See [NAVTEXT_SAMPLING.md](../NAVTEXT_SAMPLING.md#12-comparability-and-the-label-leak).',
        '- Repeated evaluation filenames/trajectory-export reruns are retained as separate artifacts, not independent seeds. Shared clip IDs do not imply shared timestamps or prompts.',
        '- The KAVA directory was reused: checkpoint-7191 belongs to a historical three-epoch run; checkpoint-2397 belongs to the later one-epoch rerun. An `e3_` evaluation filename does not prove three epochs.',
        '- Raw configs/logs can disagree. Completed training logs take precedence where recovered; CSV source columns identify the evidence. Five trained-model sweep protocols use a sibling config with checkpoint identity verified in the launch log. Two teacher camera references lack recoverable dataset configs and are explicitly marked unknown.', '',
        '## Datasets', '',
        'All identified training/evaluation datasets are LCDrive subsets of PhysicalAI-AV. The original train manifest has 38,340 clips; the KAVA teacher cache covers 38,336 of them (cache coverage is not necessarily dataset length). The navigation training file contains 50,000 event anchors, not 50,000 unique clips. The usual eval subset is 1,000 clips; KAVA used 500 evaluated clips from the validation manifest.', '',
        'Dataset IDs include cameras, timestamp selection and prompt flags. Full absolute manifest paths and dataset classes are in both CSVs.', '',
        '| ID | Clip manifest | Annotation file | Chunks | Cameras | Timing | Route slot | Camera/frame IDs |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for key, identifier in registry.items():
        dataset = dict(zip(DATASET_KEYS, key))
        lines.append(f'| {identifier} | {Path(dataset["clip_filter"]).name or "unknown"} | '
                     f'{Path(dataset["annotations"]).name or "none"} | {dataset["chunks"]} | '
                     f'{dataset["cameras"]} | {dataset["timing"]} | {dataset["route_in_prompt"]} | '
                     f'{dataset["camera_ids"]}/{dataset["frame_numbers"]} |')
    lines.extend(['', 'Route slot means `route` occurs in the prompt component order; `_nonav` annotation files intentionally leave its content empty. Repeated-looking dataset rows may differ in dataset class; see the CSV.', ''])
    headline = ['## Matched 4B Loss Ablation', '',
                'One training epoch on the original LCDrive train split (38,340 clips); 1,000 default-keyframe validation clips, four cameras, full frozen teacher expert. This is the cleanest loss-only comparison. All are no-CoT student runs. Teacher is a reference, not a trained arm.', '',
                '| Objective | ADE | minADE | Result JSON |', '|---|---|---|---|']
    selected = [('Teacher reference', 'stitch_4b_teacher.json'), ('CE', 'stitch_4b_ce.json'),
                ('CE + logit KD', 'stitch_4b_kd.json'), ('CE + logit KD + KV', 'stitch_4b_kv.json'),
                ('CE + KV', 'stitch_4b_cekv.json'), ('KV only', 'stitch_4b_kvonly.json'),
                ('KV only, depth-banded', 'stitch_4b_kvband_checkpoint-1598.json'),
                ('Block, t=0', 'stitch_4b_blockonly_checkpoint-1598.json'),
                ('Block, random t', 'stitch_4b_blockrandt_checkpoint-1598.json')]
    by_name = {Path(row['result_path']).name: row for row in evaluations}
    for label, filename in selected:
        row = by_name[filename]
        headline.append(f'| {label} | {row["ade"]:.4f} | {row["min_ade"]:.4f} | `{filename}` |')
    headline.extend(['', 'On this expert endpoint, block matching outperforms direct KV matching; logit KD does not improve the CE control. Do not transfer that ranking to the token head. The full tables retain the token-head reversal, later epochs, 2B field/span/layer-mix studies, KAVA controls, and endpoint/GT-weight/NFE sweeps.', ''])
    insert_at = lines.index('## Datasets')
    lines[insert_at:insert_at] = headline
    lines.extend(['', '## Training Directory Index', '',
                  'Loss weights below are the recorded active objective settings. The CSV also retains the full loss config, model source, initialization/resume path, learning rate, and LR multipliers. EoS trains the action expert on a frozen student VLM and is a supervised control, not another VLM-KD loss.', ''])
    for group in sorted({row['family'] for row in eligible}):
        lines.extend([f'### {group}', '', '| Run | Directory | Objective | Epoch reached | Train data | Eval files |',
                      '|---|---|---|---|---|---|'])
        for row in eligible:
            if row['family'] != group:
                continue
            lines.append(f'| {ids[row["run_id"]]} | `{Path(row["run_dir"]).name}` | {row["objective"]} | '
                         f'{row["epoch"]:.4g} | {registry[dataset_key(row)]} | {row["eval_count"]} |')
        lines.append('')
    lines.extend(['## Saved Evaluation Results', '',
                  'Every included evaluation is listed, including intermediate checkpoints >=1 epoch and controlled inference changes. Refer to the JSON filename for the specific variant; use the CSV for exact checkpoint and expert sources. `Txx` refers to the training-directory index above.', ''])
    for group in sorted({row['family'] for row in eligible}):
        group_ids = {row['run_id'] for row in eligible if row['family'] == group}
        lines.extend([f'### {group}', '', '| Run | Epoch | Eval data | Head | NFE | n | ADE | minADE | Result JSON |',
                      '|---|---|---|---|---|---|---|---|---|'])
        for row in evaluations:
            if row['status'] != 'included' or row['run_id'] not in group_ids:
                continue
            epoch = f'{row["epoch"]:.4g}' if isinstance(row['epoch'], (int, float)) else '?'
            nfe = str(row['nfe']).split(' ')[0] if row['head'] != 'token' else '-'
            lines.append(f'| {ids[row["run_id"]]} | {epoch} | {registry[dataset_key(row)]} | {row["head"]} | '
                         f'{nfe} | {row["n"]} | {row["ade"]:.4f} | {row["min_ade"]:.4f} | '
                         f'`{Path(row["result_path"]).relative_to(ROOT)}` |')
        lines.append('')
    lines.extend(['## Teacher And Baseline References', '',
                  'No local training dataset is asserted for released teacher weights. These references are not counted as training runs.', '',
                  '| Eval data | Head | NFE | n | ADE | minADE | Result JSON |', '|---|---|---|---|---|---|---|'])
    for row in evaluations:
        if row['status'] == 'teacher / baseline reference':
            nfe = str(row['nfe']).split(' ')[0] if row['head'] != 'token' else '-'
            lines.append(f'| {registry[dataset_key(row)]} | {row["head"]} | {nfe} | {row["n"]} | '
                         f'{row["ade"]:.4f} | {row["min_ade"]:.4f} | `{Path(row["result_path"]).relative_to(ROOT)}` |')
    lines.extend(['', '## Exclusions And Evidence Gaps', '',
                  '- Original root configs and logs are untouched. Full exclusions remain in the CSVs with status/reason, but are omitted from the study tables above.',
                  '- Historical pre-fix runs are retained as history, not certified as clean experiments. In particular, use the `clean_maskfix` navigation runs as the fixed baseline and consult [COMPARE_EVAL.md](../COMPARE_EVAL.md) for protocol retractions.',
                  '- `structinit_*` files are post-training weight-edit diagnostics, not separately completed training runs.', '',
                  '| Excluded training directory | Observed epoch | Reason |', '|---|---|---|'])
    for row in training:
        if row['status'] != 'included':
            lines.append(f'| `{Path(row["run_dir"]).name}` | {row["epoch"]:.4g} | {row["status"]} |')
    lines.extend(['', '### Source Warnings', ''])
    lines.extend(f'- {issue}' for issue in ISSUES)
    lines.extend(['', '## Files And Regeneration', '',
                  '- [training_runs.csv](training_runs.csv): one record per retained or excluded training history.',
                  '- [evaluations.csv](evaluations.csv): all discovered per-clip evaluations, including exclusions and source paths.',
                  '- [audit_issues.json](audit_issues.json): malformed/unreadable evidence.', '',
                  '- [completed_log_audit.csv](completed_log_audit.csv): completion-log coverage, including smoke jobs reporting >1 epoch.', '',
                  'From the repository root:', '', '```bash',
                  'recipes/alpamayo1_5_sft/.venv/bin/python recipes/alpamayo1_5_distill/scripts/summarize_ablation_runs.py', '```', ''])
    (destination / 'README.md').write_text('\n'.join(lines))


def main():
    history = configuration_history()
    runs = training_runs(history)
    configs = evaluation_configs(history)
    paths = list(ROOT.glob('*.json'))
    paths.extend(path for path in ROOT.glob('*/*.json')
                 if not path.parent.name.startswith(('output_', 'teacher_', '.stale')))
    results = [summary for path in sorted(paths)
               if (summary := summarize_metrics(path))]
    reference = next(row for row in results if Path(row['result_path']).name == 'stitch_4b_kvonly.json')
    assert round(reference['ade'], 4) == 4.1085, reference
    assert round(reference['min_ade'], 4) == 2.6313, reference
    assert reference['n'] == reference['unique_clips'] == 1000, reference
    run_index = {row['run_dir']: row for row in runs if row['run_id'] != 'kava_historical_3epoch_7191'}
    run_index.update({row['run_id']: row for row in runs})
    evaluations = [evaluated_record(row, configs, run_index) for row in results]
    training = []
    for run in runs:
        config = run['config']
        model = config.get('model', {})
        training.append({key: value for key, value in {
            **run, 'status': 'included' if run['epoch'] >= 0.999 else 'excluded: below one epoch / no completion evidence',
            'family': family(config), 'loss_config': objective(config),
            'objective': loss_label(config),
            'model_target': model.get('_target_', ''),
            'vlm': model.get('vlm_name_or_path', ''),
            'initial_checkpoint': checkpoint_from(config),
            'resume_from': config.get('trainer', {}).get('resume_from_checkpoint', ''),
            'learning_rate': config.get('trainer', {}).get('learning_rate', ''),
            'lr_multipliers': json.dumps(config.get('trainer', {}).get('lr_multiplier', {})),
            **dataset_fields(config.get('data', {}).get('train_dataset', {})),
            'eval_count': sum(row['run_id'] == run['run_id'] and row['status'] == 'included' for row in evaluations),
        }.items() if key != 'config'})
    destination = RECIPE / 'ablation_inventory'
    destination.mkdir(exist_ok=True)
    write_csv(destination / 'training_runs.csv', training)
    write_csv(destination / 'evaluations.csv', evaluations)
    (destination / 'audit_issues.json').write_text(json.dumps(ISSUES, indent=2) + '\n')
    completion_audit = []
    for path, config, epoch in completed_training_logs():
        trainer = config.get('trainer', {})
        max_steps = trainer.get('max_steps', -1)
        smoke = 0 < max_steps <= 30 and trainer.get('save_strategy') == 'no'
        directory = config.get('paths', {}).get('output_dir', '')
        accounted = any(row['run_dir'] == directory and row['epoch'] >= epoch - 0.001
                        and row['status'] == 'included' for row in training)
        assert smoke or accounted or epoch < 1, (path, epoch)
        completion_audit.append({'log_path': str(path), 'run_dir': directory, 'reported_epoch': epoch,
                                 'max_steps': max_steps, 'save_strategy': trainer.get('save_strategy', ''),
                                 'status': 'excluded: explicit tiny-data smoke' if smoke else 'covered by training inventory'})
    write_csv(destination / 'completed_log_audit.csv', completion_audit)
    render_report(destination, training, evaluations)
    print(json.dumps({'training_count': len(training),
                      'included_training': sum(row['status'] == 'included' for row in training),
                      'eval_statuses': {status: sum(row['status'] == status for row in evaluations)
                                        for status in sorted({row['status'] for row in evaluations})},
                      'unmatched': [Path(row['result_path']).name for row in evaluations
                                    if row['status'] == 'reference / unresolved training match'],
                      'issues': ISSUES}, indent=2))


if __name__ == '__main__':
    main()