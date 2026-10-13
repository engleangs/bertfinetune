"""CPU-only final evaluation from saved evidence; no training or test tuning.

The team notebook's same-gold scoring and transition analyses are retained.
Shared scoring comes from analysis.py. LODO coverage uses each Phase 2 audit,
not a Phase 1 proxy. Established BERT/loss methods are attributed in the deck
and REPRODUCIBILITY_AND_CONTRIBUTIONS.md; this module is project analysis code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd

try:
    from .analysis import align_gold, counts_to_scores, jsonl, score_records
    from .compare_results import verify_bundle
except ImportError:
    from analysis import align_gold, counts_to_scores, jsonl, score_records
    from compare_results import verify_bundle

SEEDS = (13, 42, 123, 777, 2024)
DOMAINS = ('coursera', 'food', 'hotel', 'laptop', 'phone', 'restaurant', 'sight')
MODELS = ('standard', 'weighted', 'mixed')
TRIAL = 'mixed_025_da1a88a51a5f'
ERRORS = ('correct', 'term', 'category', 'sentiment')
BANDS = ('unseen', 'rare', 'low', 'medium', 'frequent')
NULL_ON = 'focal_ia_attention_e6801d313247'
NULL_OFF = 'null_off_e53896d9134f'


def first_failure(triple, predictions):
    """Ordered gold-side outcome, not a cause or a false-positive taxonomy."""
    if triple in predictions:
        return 'correct'
    if not any(p[0] == triple[0] for p in predictions):
        return 'term'
    if not any(p[:2] == triple[:2] for p in predictions):
        return 'category'
    return 'sentiment'


def frequency_band(count):
    if count < 0:
        raise ValueError('Training counts cannot be negative')
    return ('unseen' if count == 0 else 'rare' if count <= 5 else
            'low' if count <= 20 else 'medium' if count <= 100 else 'frequent')


def vocabulary_counts(records, categories):
    """Use the evaluated model's actual source-training vocabulary."""
    known = set(categories)
    counts = Counter()
    for record in records:
        gold = set(map(tuple, record['gold_triplets']))
        predicted = set(map(tuple, record['predicted_triplets']))
        counts['predictions_outside_source_vocab'] += sum(p[1] not in known for p in predicted)
        for triple in gold:
            scope = 'covered' if triple[1] in known else 'uncovered'
            counts[scope + '_gold'] += 1
            counts[scope + '_tp'] += triple in predicted
    return {key: counts[key] for key in ('covered_gold', 'uncovered_gold', 'covered_tp',
                                         'uncovered_tp', 'predictions_outside_source_vocab')}


class EvidenceBundle:
    """Read a phase2 folder or one ZIP containing it, without extracting files."""
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.outer = None
        self.prefix = ''
        if self.path.is_dir():
            if not (self.path / 'bundle_manifest.json').is_file():
                raise ValueError('Select the phase2 folder, not the repository root')
        elif zipfile.is_zipfile(self.path):
            self.outer = zipfile.ZipFile(self.path)
            candidates = [name for name in self.outer.namelist()
                          if (name == 'bundle_manifest.json' or name.endswith('/bundle_manifest.json'))
                          and name[:-len('bundle_manifest.json')]+'comparison/in_domain_comparison.csv' in self.outer.namelist()]
            if len(candidates) != 1:
                self.outer.close()
                raise ValueError('ZIP must contain one unambiguous phase2 bundle')
            self.prefix = candidates[0][:-len('bundle_manifest.json')]
        else:
            raise ValueError(f'Phase 2 folder or ZIP not found: {self.path}')

    def read(self, name):
        member = PurePosixPath(name)
        if member.is_absolute() or '..' in member.parts or '\\' in name or ':' in name:
            raise ValueError(f'Unsafe evidence path: {name}')
        if self.outer:
            return self.outer.read(self.prefix + name)
        return (self.path / name).read_bytes()

    def data(self, name):
        return json.loads(self.read(name))

    def csv(self, name):
        return pd.read_csv(BytesIO(self.read(name)))

    def archive(self, name):
        return zipfile.ZipFile(BytesIO(self.read(name))) if self.outer else zipfile.ZipFile(self.path / name)

    def verify(self):
        if not self.outer:
            return verify_bundle(self.path)
        manifest = self.data('bundle_manifest.json')
        for entry in manifest['files']:
            content = self.read(entry['path'])
            if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError(f"Bundle input changed: {entry['path']}")
        return len(manifest['files'])

    def close(self):
        if self.outer:
            self.outer.close()


class FinalEvaluationStudy:
    """One reusable analysis behind the notebook and the command-line report."""
    def __init__(self, bundle, output=None):
        self.bundle = EvidenceBundle(bundle)
        default = self.bundle.path / 'comparison/notebook' if self.bundle.path.is_dir() else Path.cwd() / 'final_evaluation'
        self.output = Path(output).resolve() if output else default
        self.tables = {}
        self.records = {}
        self.rarity_rows = []
        self.ready = False

    @staticmethod
    def _index(rows):
        indexed = {(row['domain'], row['example_id']): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('Duplicate example identity in saved predictions')
        return indexed

    def _rarity(self, rows, counts, seed, domain):
        bands = {band: Counter() for band in BANDS}
        for row in rows:
            predicted = set(map(tuple, row['predicted_triplets']))
            for triple in set(map(tuple, row['gold_triplets'])):
                band = frequency_band(counts.get(triple[1], 0))
                bands[band]['gold'] += 1
                bands[band][first_failure(triple, predicted)] += 1
        for band, values in bands.items():
            assert sum(values[e] for e in ERRORS) == values['gold']
            self.rarity_rows.append({'scope': 'indomain' if domain == 'all_domains' else 'lodo',
                                     'domain': domain, 'seed': seed, 'rarity': band,
                                     'gold': values['gold'], **{e: values[e] for e in ERRORS}})

    def _in_domain(self, p1, p2):
        scores, alignment, duplicates, mechanisms = [], [], [], []
        reference = None
        for seed in SEEDS:
            stem = f'experiments_v2/research/indomain/all_domains/{TRIAL}/seed_{seed}/'
            current = jsonl(p2, stem + 'test_predictions.jsonl')
            audit = json.loads(p2.read(stem + 'audit.json'))
            indexed = self._index(current)
            assert len(indexed) == 3794
            gold_sets = {key: set(map(tuple, row['gold_triplets'])) for key, row in indexed.items()}
            if reference is None:
                reference = gold_sets
            assert gold_sets == reference, 'Gold differs across seeds'
            runs = {'mixed': indexed}
            categories = {'mixed': audit['categories']}
            for model in ('standard', 'weighted'):
                old_stem = f'indomain_runs/{model}/seed_{seed}/'
                legacy = jsonl(p1, old_stem + 'test_predictions.jsonl')
                old = self._index(legacy)
                runs[model] = self._index(align_gold(legacy, current))
                categories[model] = json.loads(p1.read(old_stem + 'manifest.json'))['category_vocab']
                alignment.append({'seed': seed, 'baseline': model, 'aligned_examples': len(old),
                                  'original_gold': sum(len(r['gold_triplets']) for r in old.values()),
                                  'same_gold': sum(len(g) for g in gold_sets.values())})
            self.records[seed] = runs
            self._rarity(current, audit['source_category_counts'], seed, 'all_domains')
            for model in MODELS:
                rows = list(runs[model].values())
                result = score_records(rows, categories[model])
                assert result['explicit']['gold'] == 3728
                scores.append({'seed': seed, 'model': model, **result['explicit']})
                mechanisms.append({'seed': seed, 'model': model, **result['taxonomy']})
                duplicates.append({'seed': seed, 'model': model,
                                   'extra_prediction_duplicates': sum(len(r['predicted_triplets']) -
                                    len(set(map(tuple, r['predicted_triplets']))) for r in rows)})
        self.scores = pd.DataFrame(scores)
        self.tables['in_domain'] = (self.scores.groupby('model').agg(
            precision_pct=('precision', lambda v: v.mean()*100), recall_pct=('recall', lambda v: v.mean()*100),
            f1_pct=('f1', lambda v: v.mean()*100), f1_sd_pp=('f1', lambda v: v.std()*100),
            mean_tp=('tp', 'mean'), mean_fp=('fp', 'mean'), mean_fn=('fn', 'mean')).reindex(MODELS))
        expected = self.bundle.csv('comparison/csv_reports/vs/summary.csv')
        expected = expected[(expected['split'] == 'test') & (expected['mode'] == 'indomain')].set_index('baseline')
        assert len(expected) == 2
        for model in ('standard', 'weighted'):
            assert np.isclose(self.tables['in_domain'].loc[model, 'f1_pct']/100,
                              expected.loc[model, 'explicit_f1_phase1_mean'], atol=1e-12)
        assert np.allclose(expected['explicit_f1_phase2_mean'], self.scores.query("model == 'mixed'")['f1'].mean())
        self.tables['alignment'] = pd.DataFrame(alignment)
        self.tables['duplicates'] = pd.DataFrame(duplicates)
        self.tables['taxonomy'] = pd.DataFrame(mechanisms).groupby('model')[list(ERRORS)].mean().reindex(MODELS)
        self._transitions()

    def _transitions(self):
        rows, detail = [], []
        for seed, runs in self.records.items():
            for baseline in ('standard', 'weighted'):
                counts, matrix = Counter(), Counter()
                for key, row in runs['mixed'].items():
                    gold = set(map(tuple, row['gold_triplets']))
                    base = set(map(tuple, runs[baseline][key]['predicted_triplets']))
                    mixed = set(map(tuple, row['predicted_triplets']))
                    counts.update(recovered=len((gold-base) & mixed), persistent_miss=len(gold-base-mixed),
                                  new_miss=len((gold & base)-mixed), correct_both=len(gold & base & mixed),
                                  eliminated_fp=len((base-gold)-mixed), persistent_fp=len((base-gold) & (mixed-gold)),
                                  new_fp=len((mixed-gold)-base))
                    for triple in gold:
                        matrix[first_failure(triple, base), first_failure(triple, mixed)] += 1
                original = self.scores.query('seed == @seed and model == @baseline').iloc[0]
                selected = self.scores.query("seed == @seed and model == 'mixed'").iloc[0]
                assert counts['correct_both'] + counts['new_miss'] == original.tp
                assert counts['correct_both'] + counts['recovered'] == selected.tp
                assert counts['eliminated_fp'] + counts['persistent_fp'] == original.fp
                assert counts['new_fp'] + counts['persistent_fp'] == selected.fp
                counts['delta_tp'] = selected.tp-original.tp
                counts['delta_fp'] = selected.fp-original.fp
                counts['gain_pp'] = 100*(selected.f1-original.f1)
                rows.append({'seed': seed, 'baseline': baseline, **counts})
                detail.extend({'seed': seed, 'baseline': baseline, 'before': before, 'after': after,
                               'count': matrix[before, after]} for before in ERRORS for after in ERRORS)
        self.tables['transitions_by_seed'] = pd.DataFrame(rows)
        self.tables['transitions'] = self.tables['transitions_by_seed'].groupby('baseline').agg(
            recovered=('recovered', 'mean'), newly_missed=('new_miss', 'mean'), persistent_miss=('persistent_miss', 'mean'),
            correct_both=('correct_both', 'mean'), eliminated_fp=('eliminated_fp', 'mean'), new_fp=('new_fp', 'mean'),
            delta_tp=('delta_tp', 'mean'), delta_fp=('delta_fp', 'mean'), gain_pp=('gain_pp', 'mean'),
            gain_sd_pp=('gain_pp', 'std'))
        self.tables['error_transitions'] = pd.DataFrame(detail).groupby(['baseline', 'before', 'after'])['count'].mean().unstack()

    def _lodo(self, p1, p2):
        rows = []
        baseline_rows = []
        expected = self.bundle.csv('comparison/lodo_same_gold.csv')
        assert len(expected) == 21 and (expected['n'] == 5).all()
        for domain in DOMAINS:
            for seed in SEEDS:
                stem = f'experiments_v2/research/crossdomain/{domain}/{TRIAL}/seed_{seed}/'
                audit = json.loads(p2.read(stem+'audit.json'))
                manifest = json.loads(p2.read(stem+'manifest.json'))['identity']
                assert manifest['held_out_domain'] == domain
                assert set(manifest['source_domains']) == set(DOMAINS)-{domain}
                current = jsonl(p2, stem+'test_predictions.jsonl')
                assert all(row['domain'] == domain for row in current)
                self._index(current)
                result = score_records(current, audit['categories'])
                saved = json.loads(p2.read(stem+'test_metrics.json'))['explicit']
                for ours, theirs in [('tp', 'true_positives'), ('fp', 'false_positives'), ('fn', 'false_negatives')]:
                    assert result['explicit'][ours] == saved[theirs]
                coverage = vocabulary_counts(current, audit['categories'])
                assert coverage['predictions_outside_source_vocab'] == 0
                assert coverage['uncovered_tp'] == 0
                old = json.loads(p1.read(f'lodo_runs/{domain}/standard/seed_{seed}/manifest.json'))
                rows.append({'domain': domain, 'seed': seed, **result['explicit'], **coverage,
                             'coverage_pct': 100*result['coverage'],
                             'term_f1_pct': 100*result['term']['f1'],
                             'term_category_f1_pct': 100*result['term_category']['f1'],
                             'term_sentiment_f1_pct': 100*result['term_sentiment']['f1'],
                             'vocabulary_differs_from_phase1': set(audit['categories']) != set(old['category_vocab'])})
                baseline_rows.append({'domain': domain, 'seed': seed, 'model': 'phase2_mixed_025', 'f1': result['explicit']['f1']})
                for model in ('standard', 'weighted'):
                    original_stem = f'lodo_runs/{domain}/{model}/seed_{seed}/'
                    legacy = jsonl(p1, original_stem+'test_predictions.jsonl')
                    vocabulary = json.loads(p1.read(original_stem+'manifest.json'))['category_vocab']
                    rescored = score_records(align_gold(legacy, current), vocabulary)
                    baseline_rows.append({'domain': domain, 'seed': seed, 'model': 'phase1_'+model, 'f1': rescored['explicit']['f1']})
                self._rarity(current, audit['source_category_counts'], seed, domain)
        self.tables['lodo_by_seed'] = pd.DataFrame(rows)
        frame = self.tables['lodo_by_seed']
        assert len(frame) == 35 and not frame.duplicated(['domain', 'seed']).any()
        assert (frame.covered_gold+frame.uncovered_gold == frame.gold).all()
        self.tables['lodo'] = frame.groupby('domain').agg(
            gold=('gold', 'first'), covered_gold=('covered_gold', 'first'), uncovered_gold=('uncovered_gold', 'first'),
            coverage_pct=('coverage_pct', 'mean'), exact_f1_pct=('f1', lambda v: 100*v.mean()),
            term_f1_pct=('term_f1_pct', 'mean'), term_category_f1_pct=('term_category_f1_pct', 'mean'),
            term_sentiment_f1_pct=('term_sentiment_f1_pct', 'mean')).reindex(DOMAINS)
        baselines = pd.DataFrame(baseline_rows)
        assert len(baselines) == 105 and not baselines.duplicated(['domain', 'seed', 'model']).any()
        comparison = baselines.groupby(['domain', 'model'])['f1'].mean().unstack()*100
        self.tables['lodo_baselines'] = comparison.reindex(DOMAINS).join(self.tables['lodo'][['coverage_pct']])
        for row in expected.itertuples(index=False):
            assert np.isclose(comparison.loc[row.held_out_domain, row.model]/100, row.explicit_f1, atol=1e-12)
        assert self.tables['lodo']['gold'].sum() == 3728
        for domain in DOMAINS:
            row = expected[(expected.held_out_domain == domain) & (expected.model == 'phase2_mixed_025')].iloc[0]
            assert np.isclose(self.tables['lodo'].loc[domain, 'exact_f1_pct']/100, row.explicit_f1, atol=1e-12)
            assert np.isclose(self.tables['lodo'].loc[domain, 'coverage_pct']/100, row.source_category_coverage, atol=1e-12)

    def _summarise_rarity(self):
        self.tables['rarity_by_seed'] = pd.DataFrame(self.rarity_rows)
        frame = self.tables['rarity_by_seed']
        for scope in ('indomain', 'lodo'):
            selected = frame[frame.scope == scope]
            assert (selected.groupby('seed')['gold'].sum() == 3728).all()
            # Pool domains within each seed before averaging; domains share a seed.
            per_seed = selected.groupby(['seed', 'rarity'])[['gold', *ERRORS]].sum()
            summary = per_seed.groupby('rarity').mean().reindex(BANDS)
            summary['recall_pct'] = 100*summary['correct']/summary['gold'].replace(0, np.nan)
            summary['missed'] = summary.gold-summary.correct
            summary['share_of_misses_pct'] = 100*summary.missed/summary.missed.sum()
            self.tables['rarity_'+scope] = summary
        assert np.isclose(frame.query("scope == 'lodo'")['correct'].sum(), self.tables['lodo_by_seed']['tp'].sum())

    def _null(self):
        frame = self.bundle.csv('results/null_fast_track/test_results.csv')
        assert len(frame) == 20 and not frame.duplicated(['trial_id', 'seed']).any()
        counts = frame.groupby('trial_id')['seed'].nunique()
        assert sum(counts == 5) == 2 and sum(counts == 1) == 10
        assert set(counts[counts == 5].index) == {NULL_ON, NULL_OFF}
        for trial in (NULL_ON, NULL_OFF):
            assert set(frame[frame.trial_id == trial]['seed']) == set(SEEDS)
        self.tables['null_new'] = frame.groupby('trial_id').agg(
            n_seeds=('seed', 'nunique'), explicit_f1_pct=('explicit_f1', lambda v: 100*v.mean()),
            explicit_sd_pp=('explicit_f1', lambda v: 100*v.std()),
            null_f1_pct=('null_f1', lambda v: 100*v.mean()), combined_f1_pct=('combined_f1', lambda v: 100*v.mean()),
            combined_sd_pp=('combined_f1', lambda v: 100*v.std()),
            threshold=('null_threshold', 'mean'))
        self.tables['null_new']['cohort'] = np.where(self.tables['null_new'].n_seeds == 5, 'Five-seed confirmation', 'One-seed screen')
        assert (frame[frame.trial_id == NULL_ON].null_threshold == 1).all()
        paired = frame[frame.trial_id.isin([NULL_ON, NULL_OFF])].pivot(
            index='seed', columns='trial_id', values='combined_f1').reindex(SEEDS)
        assert len(paired) == 5 and not paired.isna().any().any()
        self.tables['null_paired'] = pd.DataFrame({
            'null_on_combined_pct': paired[NULL_ON]*100,
            'control_combined_pct': paired[NULL_OFF]*100,
            'gain_pp': (paired[NULL_ON]-paired[NULL_OFF])*100})
        summary = self.bundle.csv('comparison/in_domain_comparison.csv')
        self.tables['null_controls'] = summary[['study', 'role', 'n', 'explicit_f1', 'null_f1', 'combined_f1']].copy()
        self.tables['null_controls'][['explicit_f1', 'null_f1', 'combined_f1']] *= 100
        self.tables['null_controls'].rename(columns={c: c+'_pct' for c in ('explicit_f1', 'null_f1', 'combined_f1')}, inplace=True)
        curves = []
        for seed in SEEDS:
            metric = self.bundle.data(f'evidence/null_diagnostics/{NULL_ON}/seed_{seed}/dev_metrics.json')
            for row in metric['null_threshold_curve']:
                curves.append({'seed': seed, 'threshold': row['threshold'], 'null_f1_pct': 100*row['null']['micro_f1'],
                               'combined_f1_pct': 100*row['combined']['micro_f1'],
                               'tp': row['null']['true_positives'], 'fp': row['null']['false_positives']})
        self.tables['null_development_curves'] = pd.DataFrame(curves)

    def _examples(self):
        examples = {kind: [] for kind in (*ERRORS, 'false_positive')}
        for key, row in sorted(self.records[13]['mixed'].items()):
            gold, predicted = set(map(tuple, row['gold_triplets'])), set(map(tuple, row['predicted_triplets']))
            for triple in sorted(gold):
                kind = first_failure(triple, predicted)
                if len(examples[kind]) < 2:
                    examples[kind].append({'outcome': kind, 'domain': key[0], 'example_id': key[1],
                                           'sentence': row['sentence'], 'gold': str(triple), 'predicted': str(sorted(predicted))})
            for triple in sorted(predicted-gold):
                if len(examples['false_positive']) < 2:
                    examples['false_positive'].append({'outcome': 'false_positive', 'domain': key[0], 'example_id': key[1],
                                                       'sentence': row['sentence'], 'gold': str(sorted(gold)), 'predicted': str(triple)})
        self.tables['examples'] = pd.DataFrame([row for values in examples.values() for row in values])

    def analyse(self):
        if self.ready:
            return self
        try:
            checked = self.bundle.verify()
            with ExitStack() as stack:
                p1 = stack.enter_context(self.bundle.archive('evidence/phase1_outputs.zip'))
                p2 = stack.enter_context(self.bundle.archive('evidence/phase2_outputs.zip'))
                self._in_domain(p1, p2)
                self._lodo(p1, p2)
            self._summarise_rarity()
            self._null()
            self._examples()
            self.validation = {'snapshot_files_verified': checked, 'in_domain_runs': 15, 'lodo_runs': 35,
                               'lodo_baseline_runs_rescored': 70,
                               'new_null_runs': 20, 'seeds': list(SEEDS), 'eligible_explicit_gold': 3728,
                               'phase2_covered_lodo_gold': int(self.tables['lodo'].covered_gold.sum()),
                               'phase2_uncovered_lodo_gold': int(self.tables['lodo'].uncovered_gold.sum()),
                               'lodo_vocabulary_source': 'Each evaluated Phase 2 audit.json',
                               'training_or_test_tuning_performed': False}
            self.ready = True
            return self
        finally:
            self.bundle.close()

    def show(self, section):
        """Small notebook cells display a named, already-validated table."""
        self.analyse()
        from IPython.display import display
        if section == 'validation':
            display(pd.DataFrame({'check': self.validation.keys(), 'value': [str(v) for v in self.validation.values()]}))
        elif section == 'null_new':
            display(self.tables[section].query('n_seeds == 5').round(3))
        else:
            display(self.tables[section].round(3))

    def figures(self):
        self.analyse()
        import matplotlib.pyplot as plt
        self.output.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
        paths = []
        def save(fig, name):
            fig.tight_layout()
            path = self.output / name
            fig.savefig(path, dpi=150, bbox_inches='tight')
            paths.append(path)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        table = self.tables['in_domain']
        bars = ax.bar(['Phase 1 standard', 'Phase 1 weighted', 'Phase 2 mixture'], table.f1_pct,
                      yerr=table.f1_sd_pp, capsize=5, color=['#7B8FA1', '#C49A6C', '#348B79'])
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in table.f1_pct], padding=8)
        ax.set(ylabel='Exact-triplet micro F1 (%)', title='Same-gold test performance: five seeds (sample SD)', ylim=(0, 48))
        save(fig, 'in_domain.png')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        transitions = self.tables['transitions'].reindex(['standard', 'weighted'])
        for ax, columns, labels, title in [
            (axes[0], ['recovered', 'newly_missed'], ['Recovered gold', 'Newly missed gold'], 'Gold transitions'),
            (axes[1], ['eliminated_fp', 'new_fp'], ['Eliminated FP', 'New FP'], 'False-positive transitions')]:
            transitions[columns].set_axis(labels, axis=1).plot.bar(ax=ax, color=['#348B79', '#C76C68'], rot=0)
            ax.set(ylabel='Mean triples per seed', title=title)
        save(fig, 'transitions.png')
        fig, ax = plt.subplots(figsize=(10, 4.6))
        self.tables['lodo'][['coverage_pct', 'exact_f1_pct']].rename(columns={
            'coverage_pct': 'Gold in source vocabulary', 'exact_f1_pct': 'Exact-triplet F1'}).plot.bar(
                ax=ax, color=['#7B8FA1', '#348B79'], rot=25)
        ax.set(ylabel='Percent', xlabel='Held-out domain', title='Phase 2 LODO: actual source-category coverage and F1', ylim=(0, 85))
        save(fig, 'lodo_coverage.png')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        curves = self.tables['null_development_curves']
        for ax, metric, title in [(axes[0], 'null_f1_pct', 'NULL-only development F1'),
                                  (axes[1], 'combined_f1_pct', 'Combined development F1')]:
            for seed, frame in curves.groupby('seed'):
                ax.plot(frame.threshold, frame[metric], label=str(seed), linewidth=1.5)
            ax.axvline(1, color='#C76C68', linestyle='--', label='Selected abstention')
            ax.set(xlabel='Development threshold', ylabel='F1 (%)', title=title)
            ax.legend(fontsize=8)
        save(fig, 'null_development_thresholds.png')
        return paths

    def show_figures(self):
        from IPython.display import Image, display
        for path in self.figures():
            display(Image(filename=str(path)))

    def export(self, plots=True):
        self.analyse()
        self.output.mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables.items():
            frame.to_csv(self.output / (name+'.csv'), index=True)
        (self.output / 'validation.json').write_text(json.dumps(self.validation, indent=2)+'\n', encoding='utf-8')
        if plots:
            self.figures()
        lines = ['# Final evaluation and error analysis', '',
                 'Completed presentation cohorts only; no training or parameter selection from tests.', '',
                 'All F1/precision/recall columns ending in pct are percentages; SD is across five seeds.', '']
        for name, title in [('in_domain', 'Same-gold in-domain test performance'),
                            ('transitions', 'Mean prediction transitions per seed'),
                            ('lodo_baselines', 'Same-gold LODO baseline comparison (%)'),
                            ('lodo', 'Phase 2 LODO coverage and component scores'),
                            ('rarity_indomain', 'Source-training rarity and in-domain recall')]:
            lines += ['## '+title, '']
            table = self.tables[name].round(3).reset_index()
            lines += ['| '+' | '.join(map(str, table.columns))+' |',
                      '| '+' | '.join('---' for _ in table.columns)+' |']
            lines += ['| '+' | '.join(map(str, row))+' |' for row in table.itertuples(index=False, name=None)]
            lines += ['', '']
        uncovered = self.validation['phase2_uncovered_lodo_gold']
        lines += ['## Interpretation', '', f'LODO: {uncovered}/3728 gold triples ({100*uncovered/3728:.2f}%) have categories absent from their own Phase 2 source vocabulary.', '',
                  'The newer NULL adaptation abstained; the historical BCE head recovered some pairs but reduced combined F1 against its own control.', '',
                  'These are exploratory pipeline comparisons, not a loss-only causal effect or a fresh confirmatory test.', '',
                  'The ongoing optional 120-model class-balanced extension is excluded from these completed presentation results.', '']
        if plots:
            lines += ['## Figures', '', '![Same-gold F1](in_domain.png)', '',
                      '![Prediction transitions](transitions.png)', '',
                      '![Actual Phase 2 category coverage](lodo_coverage.png)', '',
                      '![Development threshold diagnostics, not alternative test selections](null_development_thresholds.png)', '']
        (self.output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
        return self.output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    report = FinalEvaluationStudy(args.bundle, args.output)
    report.export(plots=not args.no_plots)
    print(f'Validated final evaluation: {report.output / "report.md"}')


if __name__ == '__main__':
    main()
