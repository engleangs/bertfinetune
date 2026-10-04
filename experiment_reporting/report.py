"""Self-contained HTML, concise Markdown findings and analysis exports."""

import base64
import csv
from html import escape
import json

from experiment_reporting.plots import label


def fmt(value, percentage=True, signed=False):
    if value is None:
        return "N/A"
    value = value * 100 if percentage else value
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def headline(analysis):
    ready = [r for r in analysis["comparisons"] if r.get("ready") and not r.get("smoke") and r.get("delta_f1") is not None
             and not r["verdict"].startswith("CE control")]
    if ready:
        best = max(ready, key=lambda r: r["delta_f1"])
        row = next(s for s in analysis["summaries"] if s["cohort"] == best["cohort"] and s["trial_id"] == best["trial_id"])
        return f"Best matched pilot change: {label(row)} {fmt(best['delta_f1'], signed=True)} pp versus {best['control']}. {best['verdict']}."
    if analysis["runs"] and all(r.get("smoke") for r in analysis["runs"]):
        return "Smoke checks only. These scores validate execution and do not establish a performance improvement."
    if analysis["completed_runs"]:
        return "Some runs have finished; improvement cannot yet be judged across all requested matched seeds."
    if analysis["summaries"]:
        return "Training is unfinished. Current scores are provisional; there is no completed comparison yet."
    return "No development evaluations are available yet. The report will fill in as epochs and runs finish."


def parameter_rows(analysis):
    rows = []
    for run in analysis["runs"]:
        cfg = run["config"]
        metrics = run["metrics"] or {}
        rows.append({"recipe": run["recipe"], "trial_id": run["trial_id"], "seed": run["seed"], "status": run["status"],
                     "cohort": run["cohort"], "vocabulary_scope": run["vocabulary_scope"],
                     **{key: cfg.get(key) for key in ("model_name", "model_revision", "lr", "epochs", "batch_size", "max_len",
                         "weight_decay", "warmup_ratio", "max_grad_norm", "loss_type", "bio_loss_weight", "category_loss_weight",
                         "sentiment_loss_weight", "null_head", "drop_conflict", "selection_metric")},
                     "loss_heads": ", ".join(cfg["loss_heads"]),
                     "mixture_alpha": cfg["weighted_ce_weight"] / (cfg["ce_weight"] + cfg["weighted_ce_weight"]) if cfg["loss_type"] == "mixed" else None,
                     "ce_weight": cfg["ce_weight"] if cfg["loss_type"] == "mixed" else None,
                     "weighted_ce_weight": cfg["weighted_ce_weight"] if cfg["loss_type"] == "mixed" else None,
                     "focal_gamma": cfg["focal_gamma"] if cfg["loss_type"] == "focal" else None,
                     "null_loss_weight": cfg["null_loss_weight"] if cfg["null_head"] else None,
                     "null_pos_weight_cap": cfg["null_pos_weight_cap"] if cfg["null_head"] else None,
                     "null_threshold_grid": ", ".join(map(str, cfg["null_thresholds"])) if cfg["null_head"] else None,
                     "selected_null_threshold": metrics.get("null_threshold") if cfg["null_head"] else None,
                     "best_epoch": run["best_epoch"], "epochs_recorded": run["epochs_recorded"],
                     "warmup_steps": run["training"].get("warmup_steps"),
                     "optimizer_steps": run["training"].get("optimizer_steps"),
                     "categories": run["num_categories"], "sentiments": ", ".join(run["sentiments"])})
    return rows


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_exports(analysis, output):
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "hyperparameters.csv", parameter_rows(analysis))
    write_csv(output / "scores_by_seed.csv", [
        {"recipe": r["recipe"], "trial_id": r["trial_id"], "seed": r["seed"], "cohort": r["cohort"],
         "vocabulary_scope": r["vocabulary_scope"], "status": r["status"], "provisional": r["provisional"],
         "best_epoch": r["best_epoch"], **{k: v for k, v in r["metrics"].items() if not isinstance(v, dict)}}
        for r in analysis["runs"] if r["metrics"] is not None])
    write_csv(output / "paired_changes.csv", [
        {"cohort": c["cohort"], "trial_id": c["trial_id"], "control": c["control"], "verdict": c["verdict"], **p}
        for c in analysis["comparisons"] for p in c["pairs"]])
    domains = []
    for run in analysis["runs"]:
        for domain, scores in run["per_domain"].items():
            components = scores.get("components", {})
            domains.append({"recipe": run["recipe"], "trial_id": run["trial_id"], "seed": run["seed"], "domain": domain,
                            "cohort": run["cohort"], "provisional": run["provisional"],
                            "explicit_f1": scores["explicit"]["micro_f1"],
                            "term_f1": components.get("aspect_span", {}).get("f1"),
                            "term_sentiment_f1": components.get("aspect_plus_sentiment", {}).get("f1"),
                            "category_coverage": components.get("gold_category_coverage"),
                            "gold_triplets": components.get("gold_triplets")})
    write_csv(output / "domains_by_seed.csv", domains)


def markdown_report(analysis):
    lines = ["# Development experiment comparison", "", f"Snapshot: {analysis['generated_at']} (Pacific/Auckland).", "",
             headline(analysis), "", f"Completed runs: {analysis['completed_runs']}. Planned recipe/seed keys: {analysis['expected_runs']}.", "",
             "These are development results. Provisional scores, smoke checks and different vocabulary/task/budget cohorts are separated.", "",
             "## Current scores", "", "| Recipe | Scope | Finished seeds | F1 (%) | Precision (%) | Recall (%) | Status |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in analysis["summaries"]:
        status = "Provisional best so far" if row["provisional"] else "Finished seeds only"
        deviation = row["explicit_f1_std"]
        score = fmt(row["explicit_f1_mean"]) + (f" ± {fmt(deviation)}" if deviation is not None else "")
        lines.append(f"| {row['recipe']} | {row['cohort_label'].split(' / ', 2)[-1]} | {row['completed_seeds']} | {score} | {fmt(row['precision_mean'])} | {fmt(row['recall_mean'])} | {status} |")
    lines.extend(["", "## Matched comparisons", "", "| Candidate | CE control | Paired seeds | F1 change (pp) | Interpretation |",
                  "|---|---|---|---:|---|"])
    for row in analysis["comparisons"]:
        lines.append(f"| {row['trial_id']} | {row['control']} | {[p['seed'] for p in row['pairs']]} | {fmt(row['delta_f1'], signed=True)} | {row['verdict']} |")
    lines.extend(["", "The effect target is +%.2f percentage points. Two seeds screen candidates; they do not establish significance." % (analysis['minimum_effect'] * 100),
                  "Do not compare these development scores directly with the old test-set mean: split, eligibility and protocol differ.",
                  "", "## Recorded hyperparameters", ""])
    for key, title in (("model_name", "Model"), ("lr", "Learning rate"), ("batch_size", "Batch size"), ("epochs", "Epoch budget"),
                       ("max_len", "Maximum tokens"), ("weight_decay", "Weight decay"), ("warmup_ratio", "Warmup ratio"),
                       ("max_grad_norm", "Gradient norm cap"), ("selection_metric", "Checkpoint metric")):
        values = sorted({str(r["config"][key]) for r in analysis["runs"]})
        lines.append(f"- {title}: {', '.join(values) if values else 'No recorded value yet'}")
    lines.extend(["", "See hyperparameters.csv for every run's coefficients, selected NULL threshold, best epoch and training budget.",
                  "Inactive mixture/focal/NULL parameters are N/A; the full saved config remains in analysis.json.", "",
                  "## Current taxonomy", ""])
    for row in analysis["summaries"]:
        errors = sum(row["outcomes"][name] for name in ("term", "category", "sentiment"))
        if errors:
            name = max(("term", "category", "sentiment"), key=lambda key: row["outcomes"][key])
            lines.append(f"- {row['recipe']}: {name} is the largest first-failure group, {row['outcomes'][name]}/{errors} errors ({row['outcomes'][name] / errors * 100:.2f}%)." + (" Provisional." if row["provisional"] else ""))
    lines.extend(["", "Open report.html for filters, progress, comparisons, parameters and all figures.",
                  "Rerun visualize_experiments.py to refresh, or use --watch --interval 30 while training continues."])
    if analysis["notices"]:
        lines.extend(["", "## Read notices", "", *[f"- {notice}" for notice in analysis["notices"]]])
    return "\n".join(lines) + "\n"


def html_report(analysis, figures, output):
    params = parameter_rows(analysis)
    cards = []
    for key, title in (("model_name", "Model"), ("lr", "Learning rate"), ("batch_size", "Batch size"), ("epochs", "Epoch budget"),
                       ("max_len", "Maximum tokens"), ("weight_decay", "Weight decay"), ("warmup_ratio", "Warmup ratio"),
                       ("max_grad_norm", "Gradient norm cap"), ("selection_metric", "Checkpoint metric")):
        values = sorted({str(r["config"][key]) for r in analysis["runs"]})
        cards.append(f"<div class='param'><span>{title}</span><strong>{escape(', '.join(values) or 'Pending')}</strong></div>")
    image_cards = []
    for figure in figures:
        data = base64.b64encode((output / figure["png"]).read_bytes()).decode()
        image_cards.append(f"<details open><summary>{escape(figure['title'])}</summary><img alt='{escape(figure['title'])}' src='data:image/png;base64,{data}'><p><a href='{figure['png']}'>PNG</a> · <a href='{figure['svg']}'>SVG</a></p></details>")
    payload = json.dumps({"analysis": analysis, "parameters": params}, ensure_ascii=False).replace("<", "\\u003c")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Development experiment comparison</title><style>
:root{--ink:#18354a;--muted:#64748b;--line:#dbe3ea;--teal:#168b7d;--amber:#9e641c}*{box-sizing:border-box}
body{margin:0;background:#f4f7fa;color:var(--ink);font:15px/1.6 system-ui,Arial,sans-serif}main{max-width:1320px;margin:auto;padding:34px 26px}
h1{font-size:32px;line-height:1.2;margin:12px 0}h2{font-size:21px;margin:0 0 12px}p{margin:10px 0}.muted,small{color:var(--muted)}
.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-weight:700;font-size:12px;color:var(--teal)}
.banner{background:#fff4df;border-left:4px solid #d58b27;padding:16px 20px;border-radius:8px;margin:20px 0;font-weight:600}
.section,details{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 3px 12px #18354a04}
.params{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px}.param{padding:13px;background:#f5f8fa;border-radius:8px}
.param span{display:block;color:var(--muted);font-size:12px}.param strong{font-size:17px;overflow-wrap:anywhere}
.filters{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}label{font-size:12px;color:var(--muted)}select,button{display:block;margin-top:4px;border:1px solid var(--line);padding:9px 12px;border-radius:6px;background:white;color:var(--ink)}
button{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);text-align:left;padding:9px 10px;white-space:nowrap}th{background:#f3f7fa;font-size:12px}td.number{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
.badge{display:inline-block;border-radius:20px;padding:2px 9px;font-size:11px;background:#edf2f7;color:var(--muted)}.done{background:#e7f6ee;color:#18724b}.running{background:#fff1d9;color:var(--amber)}.failed{background:#fde9e7;color:#a43b32}
img{width:100%;height:auto;display:block}summary{font-size:18px;font-weight:650;cursor:pointer}a{color:#216c9a;text-decoration:none}a:hover{text-decoration:underline}svg{width:100%;display:block}.note{font-size:13px;color:var(--muted)}
@media(max-width:650px){main{padding:22px 12px}.section,details{padding:15px}h1{font-size:27px}}
</style></head><body><main>
<div class="eyebrow">English M-ABSA · Development pilot · __DATE__</div>
<h1>Compare the experiments before the full run</h1><p class="muted">__SNAPSHOT__ · Pacific/Auckland</p>
<div class="banner">__HEADLINE__</div>
<section class="section"><h2>Progress</h2><p><strong>__COMPLETED__ finished runs</strong> · __EXPECTED__ planned recipe/seed keys · requested seeds __SEEDS__.</p>
<div class="scroll"><table id="progress"></table></div><p class="note">The status comes from saved manifests. Running scores use the best fully evaluated epoch so far; they can change. Refresh this page after regenerating the report.</p></section>
<section class="section"><h2>Current hyperparameters</h2><div class="params">__PARAMS__</div><p class="note">Recorded values from run manifests. Mixture, focal and NULL settings appear in the per-run table below. Completed checkpoints use development F1, then lower unweighted loss, then the earlier epoch.</p></section>
<section class="section"><h2>Explore scores</h2><div class="filters">
<label>Comparable cohort<select id="cohort"></select></label><label>Score<select id="metric"><option value="explicit_f1">Explicit triplet F1</option><option value="precision">Explicit precision</option><option value="recall">Explicit recall</option><option value="term_f1">Term surface F1</option><option value="boundary_f1">Exact boundary F1</option><option value="term_category_f1">Term + category F1</option><option value="term_sentiment_f1">Term + sentiment F1</option><option value="null_f1">NULL-only F1</option><option value="combined_f1">Combined F1</option><option value="macro_f1">Fixed-category macro-F1</option></select></label>
<label>Seeds<select id="seed"><option value="all">All available</option></select></label><label>&nbsp;<button onclick="location.reload()">Reload snapshot</button></label></div>
<div id="interactive"></div><div class="scroll"><table id="scores"></table></div><p class="note">Scores are percentages. Filled bars use completed seeds only; hollow bars are provisional when no seed has finished for that recipe. Whiskers show sample SD across seeds, not confidence intervals. Compare within a cohort: same task, vocabulary, budget and source data.</p></section>
<section class="section"><h2>Improvement against the CE control</h2><div class="scroll"><table id="pairs"></table></div>
<p class="note">Changes use completed, matched seed pairs. Minimum effect target: +__EFFECT__ percentage points. The NULL extension uses its vocabulary-matched head-off control. Two seeds screen promising candidates; they do not establish statistical significance. Historical test means use a different split and protocol and cannot establish improvement here.</p></section>
<section class="section"><h2>Per-run settings</h2><div class="scroll"><table id="parameters"></table></div>
<p class="note">N/A means the setting is inactive or not yet recorded. The full configuration and additional budgets are in the exports.</p></section>
__FIGURES__
<section class="section"><h2>Exports and refresh</h2><p><a href="summary.md">Written findings</a> · <a href="hyperparameters.csv">Hyperparameters</a> · <a href="scores_by_seed.csv">Per-seed scores</a> · <a href="paired_changes.csv">Matched changes</a> · <a href="domains_by_seed.csv">Domain scores and coverage</a> · <a href="analysis.json">Full snapshot</a></p>
<p class="note">Regenerate with <code>visualize_experiments.py</code>; use <code>--watch --interval 30</code> for automatic regeneration while training continues. This HTML embeds its figures and works offline. Reload to see the latest generated snapshot.</p><p class="note" id="notices"></p></section>
</main><script id="snapshot" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('snapshot').textContent),a=data.analysis;
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=(v,signed=false)=>v==null?'N/A':(signed&&v>=0?'+':'')+(100*v).toFixed(2);
const value=v=>v==null?'N/A':esc(v);
function table(id,headers,rows){document.getElementById(id).innerHTML='<thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody>'}
function badge(status){const kind=['dev_complete','complete'].includes(status)?'done':status==='running'?'running':status==='failed'?'failed':'';return '<span class="badge '+kind+'">'+esc(status)+'</span>'}
table('progress',['Recipe',...a.expected_seeds.map(s=>'Seed '+s)],a.expected_recipes.map(name=>[esc(name),...a.expected_seeds.map(seed=>{const r=a.grid.find(g=>g.recipe===name&&g.seed===seed);return badge(r.status)+(r.epochs_recorded?' · '+r.epochs_recorded+' epochs':'')})]));
const cohorts=[...new Map(a.runs.map(r=>[r.cohort,r.cohort_label+' ['+r.cohort+']'])).entries()];
document.getElementById('cohort').innerHTML=cohorts.length?cohorts.map(([id,text])=>'<option value="'+esc(id)+'">'+esc(text)+'</option>').join(''):'<option value="">Awaiting runs</option>';
[...new Set(a.runs.map(r=>r.seed))].sort((x,y)=>x-y).forEach(seed=>document.getElementById('seed').insertAdjacentHTML('beforeend','<option value="'+seed+'">Seed '+seed+'</option>'));
function render(){const cohort=document.getElementById('cohort').value,metric=document.getElementById('metric').value,seed=document.getElementById('seed').value;
const runs=a.runs.filter(r=>r.cohort===cohort&&r.metrics&&(seed==='all'||r.seed===Number(seed)));
const grouped=new Map;for(const r of runs){if(!grouped.has(r.trial_id))grouped.set(r.trial_id,[]);grouped.get(r.trial_id).push(r)}
let groups=[...grouped.values()].map(rs=>{const completed=rs.filter(r=>r.completed),chosen=completed.length?completed:rs,values=chosen.map(r=>r.metrics[metric]).filter(v=>v!=null),avg=values.length?values.reduce((s,v)=>s+v,0)/values.length:null,sd=values.length>1?Math.sqrt(values.reduce((s,v)=>s+(v-avg)**2,0)/(values.length-1)):null;return {name:rs[0].recipe,chosen,avg,sd,provisional:!completed.length}}).sort((x,y)=>(y.avg??-1)-(x.avg??-1));
const h=Math.max(120,groups.length*48+65),x0=245,width=645;
let svg='<svg viewBox="0 0 970 '+h+'" role="img" aria-label="Development scores">';
for(let tick=0;tick<=100;tick+=20){let x=x0+width*tick/100;svg+='<line x1="'+x+'" x2="'+x+'" y1="10" y2="'+(h-38)+'" stroke="#e7edf2"/><text x="'+x+'" y="'+(h-15)+'" text-anchor="middle" fill="#64748b" font-size="12">'+tick+'%</text>'}
groups.forEach((g,i)=>{let y=23+i*48,x=x0+width*(g.avg??0);svg+='<text x="0" y="'+(y+19)+'" fill="#18354a" font-size="14">'+esc(g.name)+(g.provisional?' *':'')+'</text>';if(g.avg!=null){svg+='<rect x="'+x0+'" y="'+y+'" width="'+Math.max(1,width*g.avg)+'" height="27" rx="3" stroke="'+(g.provisional?'#d58b27':'#168b7d')+'" fill="'+(g.provisional?'#fff8ec':'#168b7d')+'"/>';if(g.sd!=null)svg+='<line x1="'+(x0+width*Math.max(0,g.avg-g.sd))+'" x2="'+(x0+width*Math.min(1,g.avg+g.sd))+'" y1="'+(y+13)+'" y2="'+(y+13)+'" stroke="#18354a" stroke-width="2"/>';svg+='<text x="'+Math.min(930,x+9)+'" y="'+(y+19)+'" fill="#18354a" font-size="13">'+pct(g.avg)+'</text>'}});
if(!groups.length)svg+='<text x="300" y="50" fill="#64748b">Awaiting development evaluations</text>';
document.getElementById('interactive').innerHTML=svg+'</svg>';
table('scores',['Recipe','Shown seeds','F1 (%)','Precision (%)','Recall (%)','Best epoch','State'],groups.map(g=>{const avg=k=>g.chosen.reduce((s,r)=>s+r.metrics[k],0)/g.chosen.length;return [esc(g.name),esc(g.chosen.map(r=>r.seed).join(', ')),pct(avg('explicit_f1')),pct(avg('precision')),pct(avg('recall')),esc(g.chosen.map(r=>r.seed+': '+r.best_epoch).join(', ')),g.provisional?g.chosen.map(r=>badge(r.status)).join(' ')+' provisional':'Completed seeds']}));
table('pairs',['Candidate','CE control','Completed paired seeds','Explicit Δ (pp)','NULL Δ (pp)','Combined Δ (pp)','Interpretation'],a.comparisons.filter(c=>c.cohort===cohort).map(c=>[esc(c.trial_id),esc(c.control),esc(c.pairs.map(p=>p.seed).join(', ')||'None'),pct(c.delta_f1,true),pct(c.delta_null_f1,true),pct(c.delta_combined_f1,true),esc(c.verdict)]));
table('parameters',['Recipe','Seed','Loss','Changed heads','α','γ','BIO / category / sentiment','NULL head','NULL weight','NULL threshold','LR','Batch','Epochs','Max tokens','Best epoch'],data.parameters.filter(r=>r.cohort===cohort&&(seed==='all'||r.seed===Number(seed))).map(r=>[esc(r.recipe),r.seed,esc(r.loss_type),esc(r.loss_heads),r.mixture_alpha==null?'N/A':r.mixture_alpha.toFixed(3),value(r.focal_gamma),[r.bio_loss_weight,r.category_loss_weight,r.sentiment_loss_weight].join(' / '),r.null_head?'On':'Off',value(r.null_loss_weight),value(r.selected_null_threshold),value(r.lr),r.batch_size,r.epochs,r.max_len,value(r.best_epoch)]));}
['cohort','metric','seed'].forEach(id=>document.getElementById(id).addEventListener('change',render));render();document.getElementById('notices').textContent=a.notices.join(' | ');
</script></body></html>"""
    replacements = {"DATE": analysis["generated_at"][:10], "SNAPSHOT": analysis["generated_at"], "HEADLINE": headline(analysis),
                    "COMPLETED": analysis["completed_runs"], "EXPECTED": analysis["expected_runs"],
                    "SEEDS": ", ".join(map(str, analysis["expected_seeds"])), "EFFECT": fmt(analysis["minimum_effect"])}
    for key, text in replacements.items():
        template = template.replace(f"__{key}__", escape(str(text)))
    return template.replace("__PARAMS__", "".join(cards)).replace("__FIGURES__", "".join(image_cards)).replace("__PAYLOAD__", payload)


def write_report(analysis, figures, output):
    write_exports(analysis, output)
    (output / "summary.md").write_text(markdown_report(analysis), encoding="utf-8")
    temporary = output / ".report.html.tmp"
    temporary.write_text(html_report(analysis, figures, output), encoding="utf-8")
    temporary.replace(output / "report.html")
