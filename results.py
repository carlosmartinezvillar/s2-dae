import os
import csv
import json
import argparse

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
				f"proj_dim {r['proj_dim']} | init_chan {r['init_chan']} | lrate {r['lrate']:g} | "
				f"decay {r['decay']:g} | bt_lambda {r['bt_lambda']:g}")

	return {encoder:runs[0] for encoder,runs in ranked.items()}


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--probe-dir',required=True,help='Directory of bt_probe logs (probe_XXX.tsv, model_XXX.done).')
	parser.add_argument('--params',default='./hpo/bt_pretrain.json',help='BT pretraining params file the probe ran on.')
	parser.add_argument('--top',type=int,default=5,help='Runs to print per encoder type.')
	args = parser.parse_args()

	best = best_bt_encoders(args.probe_dir,args.params,top=args.top)
	for encoder,row in sorted(best.items()):
		print(f"Best {encoder.upper()} encoder: id {row['id']:03}")
