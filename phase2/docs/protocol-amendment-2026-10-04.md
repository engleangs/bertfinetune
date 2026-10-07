# Protocol amendment: taxonomy and staged continuation

Version: 1.1 reporting amendment. Date: 2026-10-04, Pacific/Auckland.
Status on 2026-10-04: active for analysis of existing runs; new training pending.
Original protocol: [protocol.md](protocol.md), version 1.0, 2026-09-17.

Implementation update, 2026-10-05: corrected training is now available in the
separate `run_experiments.py` runner. The NULL extension, experiment flags,
two-seed pilot and promotion rules are recorded in
[the dated experiment guide](experiment-guide-2026-10-05.md). Version-1
training/results remain historical; the statements below describe the
4 October review and proposed continuation.

## Research focus and status of the evidence

The main research contribution is a reproducible error taxonomy: where does
explicit-aspect TASD fail, and how do domain shift and class imbalance change
the error distribution? The standard-versus-weighted comparison supports
this diagnosis. A positive F1 improvement is not required for an informative
taxonomy study.

Existing evidence comprises 10 in-domain runs and 70 leave-one-domain-out
runs, using standard and weighted loss with five matched seeds. These test
results have been inspected. The new taxonomy, component metrics, loss
experiments and domain rankings are exploratory. No new claim of untouched
confirmatory evaluation is made from these same tests.

This amendment changes reporting definitions and records implementation
deviations. It does not retroactively change the old predictions or relabel
them as corrected training runs. Original version-1 tables remain available.

## 1. One primary error, with independent diagnostic flags

For each unique retained gold triplet, inspect predictions within its sentence:

| Outcome | Rule |
|---|---|
| Correct | The exact term, category and sentiment triplet is predicted |
| Term error | No prediction has the exact term surface text |
| Category error | The term is predicted, but no prediction on it has the gold category |
| Sentiment error | Term and category are predicted together, but sentiment differs |

Apply the rules in order. The three error outcomes partition incorrect gold.
Call this the **first failed component**, not the sole cause. Independent
flags retain cases in which both category and sentiment are wrong.

Count spurious predictions and duplicates separately on the prediction side.
An unsupported NULL target is a representational limitation outside the
explicit-only gold taxonomy, not automatically a term-model failure.

Report both counts and these denominators:

- Primary error share: errors of one type / all incorrect retained gold.
- Term failure rate: missed gold terms / all retained gold triplets.
- Category failure given term: primary category errors / gold with a predicted term.
- Sentiment failure given term and category: primary sentiment errors / gold
  with term and category predicted together.

An empty denominator is N/A. These conditional rates are membership
diagnostics of saved predictions, not isolated head accuracies. Missing
terms are excluded from the category denominator; missing terms and wrong
categories are excluded from the conditional sentiment denominator.

Average counts over seeds. Calculate the compact taxonomy rates from pooled
gold outcomes across those seeds, and label them accordingly. Replication
uncertainty and paired comparisons use seed-level observations; repeated
gold sentences across seeds are not independent samples for significance tests.

## 2. Full-triplet and component evaluation

Keep exact-triplet micro-F1 as the primary performance metric and checkpoint
selection metric. The taxonomy is the main diagnostic analysis. Add:

| Metric | Sentence-level matching key | What it measures |
|---|---|---|
| Term F1 | term | Extraction without label requirements |
| Term + category F1 | (term, category) | Extraction plus category |
| Term + sentiment F1 | (term, sentiment) | Category-independent extraction and sentiment |
| Full-triplet F1 | (term, category, sentiment) | Complete task performance |
| Known-category triplet F1 | Full triplet after source-known category filtering | Performance on supported categories, with coverage shown |

All use set semantics within each sentence, deduplicating after projection.
Always report precision and recall with F1. Component metrics supplement full
matching; they are not a replacement or an additive decomposition of it.

Full-triplet evaluation keeps unseen-category gold as false negatives.
Known-category F1 is N/A when no known-category gold exists. Category-free
scores can reveal transfer hidden by label mismatch but cannot solve the
model's inability to output unseen categories.

Current artifacts save surface text without occurrence offsets. Accordingly,
report **term surface F1**, not boundary F1. Future runs must save token and
character offsets before exact boundaries, partial overlap or repeated
occurrences can be evaluated. Do not reconstruct guessed offsets from text.

The existing macro-F1 over a variable gold/prediction label union is a legacy
metric. For corrected runs, freeze category macro-F1 over the source-training
category universe, assign zero to zero-denominator categories, and report
unseen categories separately while retaining their false negatives in full
micro-F1. Do not silently mix the old and corrected macro-F1 tables.

## 3. Rare, unseen and NULL scope

Category frequency comes exclusively from the fold's source-training retained
annotations. Freeze bands at unseen=0, rare=1-5, low=6-20, medium=21-100,
frequent>100. Give each band its gold support, exact-triplet recall and primary
error composition. Unseen labels must remain separate from rare labels.

For each domain and official split, count raw and deduplicated annotations,
NULL percentage, NULL-only/explicit-only/mixed/empty sentences, and NULL
category/sentiment distributions. NULL prevalence divides by all deduplicated
gold annotations, not only explicit ones. Report annotation-scope recall bound
as retained explicit gold / all deduplicated gold. Tokenization, unknown
categories and stricter eligibility may impose further limits.

Keep explicit-only F1 unchanged by the NULL audit. A full-annotation score
including NULL needs a separately labelled evaluation against all gold, with
NULL false negatives retained; prevalence alone is not that evaluation. A
NULL-aware model is a later architectural extension.

## 4. Sentiment labels and Conflict

Keep positive, negative, neutral and Conflict in the historical comparison.
The audit finds five deduplicated Conflict training annotations, one retained
explicit training target, and zero development/test Conflict gold. This limits
what can be concluded about that class.

Removing Conflict is optional, deferred sensitivity analysis. It changes the
task and must receive its own result label. Filter Conflict annotations before
rebuilding training vocabularies and weights; preserve other annotations in
the sentence. Do not relabel it neutral or alter historical scores.

## 5. Hardest component and hardest domain

Use largest primary error count/share to name the largest observed error
group. Use conditional rates to assess labels after an aspect is detected.
Do not equate either with causal head difficulty; an ablation is required.

Rank domains separately for in-domain and LODO conditions, separately by loss.
Show full-triplet F1 with category coverage and term/term+sentiment F1. A
zero-coverage fold is unidentifiable on closed-set full-triplet performance;
report ties instead of choosing an arbitrary hardest domain.

Current findings: term errors dominate standard in-domain loss; category
errors dominate weighted in-domain loss. Food has the lowest in-domain
triplet F1 under both losses. LODO full-triplet zeros tie across four domains;
phone has the lowest standard term F1, and food the lowest weighted term F1.
These are descriptive rankings, not significance claims between every pair.

## 6. Implementation deviations and corrected-training specification

| Item | Historical implementation | Required for separately versioned corrected runs |
|---|---|---|
| Warmup | Zero | floor(0.10 * total optimizer steps) |
| Gradient clipping | Absent | Clip norm to 1.0 after backward, before optimizer step |
| Repeated term alignment | First occurrence | Exclude ambiguous occurrences; count exclusions |
| Multiple distinct label pairs | Keep first pair | Exclude the entire conflicting span; count exclusions |
| Overlap eligibility | Greedy annotation order | Freeze a deterministic exclusion policy before running |
| Macro-F1 universe | Variable gold/prediction union | Fixed source-training category universe |
| Category/term/sentiment reporting | Partial post-hoc metrics | Save components and diagnostic records with each run |
| Provenance | Limited package versions | Package lock, data/model/tokenizer revisions, code commit and dirty state |

These corrections are **not yet implemented in the trainer** by this amendment.
Keep model, learning rate, batch size, five-epoch budget, weight decay,
checkpoint tie-breaks and seeds from version 1.0. Record corrected experiments
under a new experiment version and distinct output paths. Recompute audits,
vocabularies and rare sets after eligibility changes.

## 7. Professor's loss feedback: proposed staged experiment

Only after the corrected pipeline passes its gates, define per-head loss:

    L_h = (1 - alpha) * CE_h + alpha * inverse_frequency_CE_h
    L_total = L_BIO + L_category + L_sentiment

This divides the professor's w1*CE + w2*weighted_CE by w1+w2, making
alpha=w2/(w1+w2). The examples (1, 0.5) and (0.5, 0.5) become alpha=1/3 and
1/2. Document the normalization: an unnormalized sum is a different loss
scale. Freeze the current CE and weighted-CE mean reductions, train-only
class counts and outer head multipliers.

Proposed in-domain development pilot: seed 13; six mixture values
0, 0.25, 1/3, 0.5, 0.75, 1; two unweighted focal settings gamma=1 and 2.
Apply each candidate consistently across all three heads in this first sweep.
Compute focal p_t from unweighted probabilities and ignore -100 targets.
Do not add class weighting to focal in this initial candidate pool.
([Focal loss paper](https://arxiv.org/abs/1708.02002))

This is a declared next-stage budget, not a run launched by this amendment.
Select a shortlist on development full-triplet F1 and inspect precision,
recall, rarity and taxonomy as secondary diagnostics. Isolate heads in a
subsequent ablation; expand selected designs to all five matched seeds.

New LODO tuning must use only each fold's source-development splits. Do not
carry a recipe selected with held-out target-development examples into a
strict no-target-tuning claim. Fix equal candidate budgets before comparing
folds. No learning-rate/epoch search or three-class task change is combined
with the initial loss sweep.

## 8. Current work versus next work

Completed now: notebook review, historical-run validation, exclusive taxonomy,
component F1, conditional diagnostics, rarity, NULL and sentiment audits, and
this reporting amendment. Analysis implementation lives in separate loading,
diagnostic, taxonomy-summary and reporting modules.

Next: implement corrected training/eligibility/metric rules, run focused tests
and a tiny local integration check, save revised audits and provenance, then
freeze the corrected experiment version. Start the small development loss
pilot after those gates. Do not rerun the entire GPU matrix yet.
