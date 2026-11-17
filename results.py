import io
import os
import csv
import contextlib
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

################################################################################
# FUNCTIONS/HELPERS
################################################################################
def load_params(path):
	'''
	JSON Lines hyperparameter file -> {id: row}.
	'''
	with open(path,'r') as fp:
		return {row['id']:row for row in map(json.loads,filter(str.strip,fp))}


def best_probe_epoch(tsv_path,metric='iou_1',warmup=10):
	'''
	Best validation epoch of one probe log (probe_XXX.tsv), picked the same way
	train_condition.py picks the saved checkpoint: highest metric after warmup.
	train_base.py uses the same rule, so this also reads base_XXX.tsv.
	Returns that epoch's row as {column: value}, or None if no epoch passed warmup.
	'''
	with open(tsv_path,'r') as fp:
		rows = [r for r in csv.DictReader(fp,delimiter='\t') if int(r['epoch']) >= warmup]
	if not rows:
		return None

	best = max(rows,key=lambda r: float(r[metric])) # FIRST MAX ON TIES, LIKE THE STRICT > IN train_probe()
	return {k:(int(v) if k == 'epoch' else float(v)) for k,v in best.items()}


################################################################################
# BT PROBE
################################################################################
def best_bt_encoders(probe_dir,params_file='./hpo/bt_pretrain.json',metric='iou_1',top=5):
	'''
	Best BT encoder per encoder type ('cnn', 'vit') by the probe's validation metric.
	probe_dir holds the bt_probe logs (probe_XXX.tsv, model_XXX.done) copied from
	/log_pvc/bt_probe. A run counts only once its .done marker exists, so a sweep
	still in progress is reported as such instead of ranked as if complete.
	Prints the top runs per type and returns {encoder: best row}, where each row is
	the params row plus its best probe epoch under 'probe'.
	'''
	params  = load_params(params_file)
	ranked  = {}
	missing = []

	for i,hp in sorted(params.items()):
		tsv  = f'{probe_dir}/probe_{i:03}.tsv'
		done = os.path.isfile(f'{probe_dir}/model_{i:03}.done')
		best = best_probe_epoch(tsv,metric) if (done and os.path.isfile(tsv)) else None
		if best is None:
			missing.append(i)
			continue
		ranked.setdefault(hp.get('encoder','cnn'),[]).append({**hp,'probe':best})

	for runs in ranked.values():
		runs.sort(key=lambda r: r['probe'][metric],reverse=True)

	# REPORT
	if missing:
		print(f'{len(missing)} of {len(params)} probe runs not finished: {missing}')
	for encoder,runs in sorted(ranked.items()):
		print(f'--- {encoder.upper()}: top {min(top,len(runs))} of {len(runs)} runs by {metric} ---')
		for r in runs[:top]:
			print(f"id {r['id']:03} | {metric} {r['probe'][metric]:.5f} @ epoch {r['probe']['epoch']:03} | "
				f"proj_dim {r['proj_dim']} | init_chan {r['init_chan']} | vit_layers {r.get('vit_layers',1 if encoder == 'vit' else 0)} | "
				f"lrate {r['lrate']:g} | decay {r['decay']:g} | lambda_d {r['lambda_d']:g} | bt_lambda {r['bt_lambda']:g}")

	return {encoder:runs[0] for encoder,runs in ranked.items()}


################################################################################
# SUPERVISED BASELINE
################################################################################
def best_base(base_dir,params_file='./hpo/base_train.json',metric='iou_1',top=None):
	'''
	Supervised baseline runs (train_base.py) sorted by their best validation metric after
	warmup, i.e. the epoch of the saved base_XXX_e*.pth.tar checkpoint. base_dir holds the
	base_train logs (base_XXX.tsv, model_XXX.done) copied from /log_pvc/base_train; a run
	counts only once its .done marker exists. Prints the top runs (all if top is None) and
	returns the sorted list of params rows, each with its best epoch under 'base'.
	'''
	params  = load_params(params_file)
	ranked  = []
	missing = []

	for i,hp in sorted(params.items()):
		tsv  = f'{base_dir}/base_{i:03}.tsv'
		done = os.path.isfile(f'{base_dir}/model_{i:03}.done')
		best = best_probe_epoch(tsv,metric) if (done and os.path.isfile(tsv)) else None
		if best is None:
			missing.append(i)
			continue
		ranked.append({**hp,'base':best})

	ranked.sort(key=lambda r: r['base'][metric],reverse=True)

	# REPORT
	if missing:
		print(f'{len(missing)} of {len(params)} base runs not finished: {missing}')
	n = len(ranked) if top is None else min(top,len(ranked))
	print(f'--- BASE: top {n} of {len(ranked)} runs by {metric} ---')
	for rank,r in enumerate(ranked[:n],start=1):
		print(f"{rank:3} | id {r['id']:03} | {metric} {r['base'][metric]:.5f} @ epoch {r['base']['epoch']:03} | "
			f"vloss {r['base']['vloss']:.5f} | {r['encoder']}-{r['decoder']} | init_chan {r['init_chan']} | "
			f"vit_layers {r['vit_layers']} | batch {r['batch']} | lrate {r['lrate']:g} | decay {r['decay']:g}")

	return ranked


################################################################################
# PARAMETER EFFECTS
################################################################################
def base_effects(base_dir,params_file='./hpo/base_train.json',metric='iou_1',lr_max=4e-4):
	'''
	Which base_train hyperparameters influence the best validation metric, each with the others held fixed.
	Fits one linear model (ordinary least squares) of the metric on every hyperparameter at once and, for each
	one, tests how much worse the fit gets without it (partial F-test). Prints, per parameter: its effect in
	metric units, the share of the remaining variance it explains (partial R^2), F and the p-value.
	Run three times:
	  1. all runs: the learning rate dominates through the collapse of ViT models above ~5e-4 lrate;
	  2. plateau only (lrate <= lr_max): removes the collapse, so smaller effects are not swamped by it and an
	     effect that interacts with it (wider models collapse at lower lrate) is not mistaken for a main effect;
	  3. plateau, on ranks of the metric: the same test, robust to a few outlying runs.
	A parameter is credible if it is significant in 2 and 3 and its effect is large enough to matter compared
	with the residual SD (the scatter the model leaves unexplained, an upper bound on seed noise: one seed per run).
	Returns {analysis label: {parameter: dict(partial_r2, F, p, coef)}}.
	'''
	with contextlib.redirect_stdout(io.StringIO()): # REUSE THE RANKING WITHOUT ITS PRINTOUT
		ranked = best_base(base_dir,params_file,metric,top=0)
	plateau = [r for r in ranked if r['lrate'] <= lr_max]

	print(f'Reference levels: arch cnn-cnn, init_chan 32, vit_layers 1, batch 16 (coef = change vs reference, in {metric} units)')
	out = {}
	for label,runs,use_rank in (('all runs',ranked,False),
	                            (f'plateau (lrate <= {lr_max:g})',plateau,False),
	                            (f'plateau (lrate <= {lr_max:g}), ranks of {metric}',plateau,True)):

		# RESPONSE: THE METRIC, OR ITS RANKS (1 = WORST). RANKS MAKE EVERY RUN COUNT THE SAME, SO A FEW
		# COLLAPSED RUNS CANNOT DRIVE THE FIT; COEFFICIENTS ARE THEN IN RANK UNITS AND ARE NOT PRINTED.
		y = np.array([r['base'][metric] for r in runs])
		if use_rank:
			y = stats.rankdata(y)

		# PREDICTORS, ONE BLOCK OF COLUMNS PER HYPERPARAMETER. CATEGORIES ARE DUMMY CODED: A 0/1 COLUMN PER
		# NON-REFERENCE LEVEL, SO EACH COEFFICIENT IS THE DIFFERENCE TO THE REFERENCE LEVEL.
		arch   = [f"{r['encoder']}-{r['decoder']}" for r in runs]
		log_lr = np.log10([r['lrate'] for r in runs])
		log_lr = log_lr - log_lr.mean() # CENTERED, SO THE LINEAR AND SQUARED TERMS ARE LESS CORRELATED
		blocks = {
			'arch':       np.array([[a == x for x in ('cnn-vit','vit-cnn','vit-vit')] for a in arch],float),
			'init_chan':  np.array([[r['init_chan'] == 64] for r in runs],float),
			# cnn-cnn HAS NO ViT STAGE (vit_layers 0) AND ITS OWN arch LEVEL, SO THIS COLUMN COMPARES
			# DEPTH 2 WITH DEPTH 1 WITHIN THE PAIRS THAT HAVE A ViT PART
			'vit_layers': np.array([[r['vit_layers'] == 2] for r in runs],float),
			'batch':      np.array([[r['batch'] == 32] for r in runs],float),
			# LOG LRATE + ITS SQUARE: A CURVE, SO A PLATEAU THAT DROPS AT HIGH LRATE CAN BE FITTED
			'lrate':      np.column_stack([log_lr,log_lr**2]),
			'decay':      np.log10([[r['decay']] for r in runs]),
		}
		X = np.column_stack([np.ones(len(runs))] + list(blocks.values())) # FIRST COLUMN: INTERCEPT

		# FULL MODEL: LEAST-SQUARES COEFFICIENTS b AND RESIDUAL SUM OF SQUARES RSS = sum (y - Xb)^2
		b,*_   = np.linalg.lstsq(X,y,rcond=None)
		rss    = float(np.sum((y - X @ b)**2))
		df_res = len(y) - X.shape[1]        # RESIDUAL DEGREES OF FREEDOM: RUNS MINUS FITTED COEFFICIENTS
		tss    = float(np.sum((y - y.mean())**2))
		res_sd = (rss/df_res)**0.5          # TYPICAL DISTANCE OF A RUN FROM THE MODEL
		unit   = 'ranks' if use_rank else metric
		print(f'--- {label}: n={len(y)} | R^2 {1 - rss/tss:.2f} (share of variance the model explains) | residual SD {res_sd:.3g} {unit}')

		# DROP ONE PARAMETER AT A TIME (REDUCED MODEL) AND COMPARE WITH THE FULL MODEL
		out[label] = {}
		col = 1 # COLUMN 0 IS THE INTERCEPT
		for name,block in blocks.items():
			m    = block.shape[1] # COLUMNS (COEFFICIENTS) THIS PARAMETER USES
			keep = [c for c in range(X.shape[1]) if not (col <= c < col+m)]
			b_r,*_ = np.linalg.lstsq(X[:,keep],y,rcond=None)
			rss_r  = float(np.sum((y - X[:,keep] @ b_r)**2))

			# PARTIAL F: EXTRA ERROR PER DROPPED COEFFICIENT, RELATIVE TO THE FULL MODEL'S ERROR PER RESIDUAL DF.
			# F ~ 1 IF THE PARAMETER ADDS NOTHING BEYOND NOISE; p = P(F >= observed) UNDER NO EFFECT (F(m, df_res)).
			F = ((rss_r - rss)/m) / (rss/df_res)
			p = float(stats.f.sf(F,m,df_res))

			# PARTIAL R^2: SHARE OF THE VARIANCE LEFT BY THE OTHER PARAMETERS THAT THIS ONE EXPLAINS
			partial_r2 = (rss_r - rss)/rss_r

			coef = b[col:col+m]
			effect = '' if use_rank else '  coef ' + ', '.join(f'{c:+.5f}' for c in coef)
			print(f'   {name:10s} partial R^2 {partial_r2:5.3f} | F {F:6.1f} | p {p:.2g}{effect}')
			out[label][name] = {'partial_r2':partial_r2,'F':F,'p':p,'coef':coef.tolist()}
			col += m

	print("Coef order: arch = cnn-vit, vit-cnn, vit-vit; lrate = log10(lrate) (centered), its square; decay = log10(decay).")
	return out


################################################################################
# PLOTS
################################################################################
# CATEGORICAL COLORS IN FIXED ORDER (blue, orange, aqua, yellow), ONE PER encoder-decoder PAIR
ARCH_COLORS = {'cnn-cnn':'#2a78d6','cnn-vit':'#eb6834','vit-cnn':'#1baf7a','vit-vit':'#eda100'}

def plot_base_histograms(base_dir,params_file='./hpo/base_train.json',metric='iou_1',bin_width=0.001,out_path=None):
	'''
	Superimposed histograms of each base run's best validation metric (after warmup, as in best_base()),
	one per encoder-decoder pair (cnn-cnn, cnn-vit, vit-cnn, vit-vit). Shared bins of bin_width; the height
	is the share of that pair's runs (cnn-cnn has half as many runs as the others), so the shapes compare
	directly. A thin vertical line marks each pair's median. Saves to out_path if given, else shows.
	'''
	with contextlib.redirect_stdout(io.StringIO()): # REUSE THE RANKING WITHOUT ITS PRINTOUT
		ranked = best_base(base_dir,params_file,metric,top=0)

	values = {arch:[] for arch in ARCH_COLORS}
	for r in ranked:
		values[f"{r['encoder']}-{r['decoder']}"].append(r['base'][metric])

	# SHARED BIN EDGES OVER ALL RUNS
	lo = min(min(v) for v in values.values() if v)
	hi = max(max(v) for v in values.values() if v)
	n_bins = int((hi-lo)/bin_width)+2
	edges  = [lo - bin_width/2 + k*bin_width for k in range(n_bins+1)]

	fig,ax = plt.subplots(figsize=(9,4.5))
	for arch,v in values.items():
		if not v:
			continue
		color  = ARCH_COLORS[arch]
		weight = [1.0/len(v)]*len(v)
		median = sorted(v)[len(v)//2]
		ax.hist(v,bins=edges,weights=weight,color=color,alpha=0.12,linewidth=0)                    # LIGHT FILL
		ax.hist(v,bins=edges,weights=weight,color=color,histtype='step',linewidth=2,               # OUTLINE
			label=f'{arch}  (n={len(v)}, median {median:.4f})')
		ax.axvline(median,color=color,linewidth=1)

	# RECESSIVE AXES
	for side in ('top','right'):
		ax.spines[side].set_visible(False)
	for side in ('left','bottom'):
		ax.spines[side].set_color('#c3c2b7')
	ax.grid(axis='y',color='#e6e5e0',linewidth=0.8)
	ax.set_axisbelow(True)
	ax.tick_params(colors='#5f5e5a')
	ax.set_xlabel(f'best validation {metric} (after warmup)',color='#3d3d3a')
	ax.set_ylabel('share of runs',color='#3d3d3a')
	ax.set_title(f'Supervised baselines: {metric} by encoder-decoder ({len(ranked)} runs)',color='#1f1e1d',loc='left')
	ax.legend(frameon=False,loc='upper left')
	fig.tight_layout()

	if out_path:
		fig.savefig(out_path,dpi=150)
		print(f'Saved {out_path}')
	else:
		plt.show()
	plt.close(fig)


def plot_base_lrate(base_dir,params_file='./hpo/base_train.json',metric='iou_1',lr_key='lrate',out_path=None):
	'''
	Best validation metric of each base run (after warmup, as in best_base()) against its learning rate, log x.
	Small multiples, one panel per encoder-decoder pair with shared axes: the pair's runs in its color over
	all other runs in gray for context, plus its median per log-spaced lrate bin (bins with >= 3 runs).
	lr_key 'lrate' is the lr the run used; 'base_lrate' is the same at the reference batch 16
	(lrate = base_lrate * sqrt(batch/16)), comparable across batch sizes. Saves to out_path if given, else shows.
	'''
	with contextlib.redirect_stdout(io.StringIO()): # REUSE THE RANKING WITHOUT ITS PRINTOUT
		ranked = best_base(base_dir,params_file,metric,top=0)

	points = {arch:[] for arch in ARCH_COLORS}
	for r in ranked:
		points[f"{r['encoder']}-{r['decoder']}"].append((r[lr_key],r['base'][metric]))

	# LOG-SPACED LRATE BINS OVER ALL RUNS, FOR THE MEDIAN LINE
	lrs     = [lr for v in points.values() for lr,_ in v]
	lo,hi   = min(lrs),max(lrs)
	n_bins  = 8
	edges   = [lo*(hi/lo)**(k/n_bins) for k in range(n_bins+1)]
	edges[-1] *= 1.0001 # INCLUDE THE MAX

	fig,axes = plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
	for ax,(arch,v) in zip(axes.flat,points.items()):
		color = ARCH_COLORS[arch]

		# CONTEXT: EVERY OTHER PAIR IN GRAY, BEHIND
		others = [p for a,w in points.items() if a != arch for p in w]
		ax.scatter([p[0] for p in others],[p[1] for p in others],s=14,color='#d3d2cc',linewidths=0,zorder=1)

		# THIS PAIR, WITH A SURFACE RING SO OVERLAPPING MARKERS STAY LEGIBLE
		ax.scatter([p[0] for p in v],[p[1] for p in v],s=36,color=color,edgecolors='white',linewidths=1,zorder=2)

		# MEDIAN PER LRATE BIN
		mx,my = [],[]
		for k in range(n_bins):
			inbin = sorted(y for x,y in v if edges[k] <= x < edges[k+1])
			if len(inbin) >= 3:
				mx.append((edges[k]*edges[k+1])**0.5)
				my.append(inbin[len(inbin)//2])
		ax.plot(mx,my,color=color,linewidth=2,zorder=3)

		best = max(v,key=lambda p: p[1])
		ax.set_title(f'{arch}  (n={len(v)}, best {best[1]:.4f} @ {lr_key} {best[0]:.2g})',color='#1f1e1d',loc='left',fontsize=10)
		ax.set_xscale('log')

		# RECESSIVE AXES
		for side in ('top','right'):
			ax.spines[side].set_visible(False)
		for side in ('left','bottom'):
			ax.spines[side].set_color('#c3c2b7')
		ax.grid(color='#e6e5e0',linewidth=0.8)
		ax.set_axisbelow(True)
		ax.tick_params(colors='#5f5e5a',which='both')

	for ax in axes[1]:
		ax.set_xlabel(lr_key + (' (lr at batch 16)' if lr_key == 'base_lrate' else ' (lr used)'),color='#3d3d3a')
	for ax in axes[:,0]:
		ax.set_ylabel(f'best validation {metric}',color='#3d3d3a')
	fig.suptitle(f'Supervised baselines: {metric} vs {lr_key} ({len(ranked)} runs; line = median per {lr_key} bin)',color='#1f1e1d',x=0.01,ha='left')
	fig.tight_layout()

	if out_path:
		fig.savefig(out_path,dpi=150)
		print(f'Saved {out_path}')
	else:
		plt.show()
	plt.close(fig)


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--probe-dir',default=None,help='Directory of bt_probe logs (probe_XXX.tsv, model_XXX.done).')
	parser.add_argument('--params',default='./hpo/bt_pretrain.json',help='BT pretraining params file the probe ran on.')
	parser.add_argument('--base-dir',default=None,help='Directory of base_train logs (base_XXX.tsv, model_XXX.done).')
	parser.add_argument('--base-params',default='./hpo/base_train.json',help='Params file the base_train run used.')
	parser.add_argument('--top',type=int,default=None,help='Runs to print (probe: per encoder type, default 5; base: default all).')
	parser.add_argument('--base-hist',default=None,help='With --base-dir: save the iou_1 histogram per encoder-decoder to this path.')
	parser.add_argument('--base-lr',default=None,help='With --base-dir: save the iou_1 vs learning rate plot per encoder-decoder to this path.')
	parser.add_argument('--base-effects',action='store_true',help='With --base-dir: test which hyperparameters influence iou_1 (partial F-tests).')
	args = parser.parse_args()
	assert (args.base_hist is None and args.base_lr is None and not args.base_effects) or args.base_dir, "--base-hist/--base-lr/--base-effects need --base-dir"
	assert args.probe_dir or args.base_dir, "Give --probe-dir and/or --base-dir"

	if args.probe_dir:
		best = best_bt_encoders(args.probe_dir,args.params,top=5 if args.top is None else args.top)
		for encoder,row in sorted(best.items()):
			print(f"Best {encoder.upper()} encoder: id {row['id']:03}")

	if args.base_dir:
		best_base(args.base_dir,args.base_params,top=args.top)
		if args.base_hist:
			plot_base_histograms(args.base_dir,args.base_params,out_path=args.base_hist)
		if args.base_lr:
			plot_base_lrate(args.base_dir,args.base_params,out_path=args.base_lr)
		if args.base_effects:
			base_effects(args.base_dir,args.base_params)
