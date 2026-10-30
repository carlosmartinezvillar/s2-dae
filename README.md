# s2-dae
A Simple Denoising AutoEncoder for Sentinel-2 images. 

## Barlow Twins workflow (Kubernetes)

Run from the repo root. Commit and push each new params file before launching it: the pods clone the remote.

```
# 1. Pretrain every BT configuration (500 epochs each)
./k8s/kube_launcher.sh bt_pretrain ./hpo/bt_pretrain.json 0 239

# 2. Probe every encoder from the sweep (frozen encoder, decoder only)
./k8s/kube_launcher.sh bt_probe ./hpo/bt_probe.json 0 239

# 3. Pretrain the best encoder from the probe for 1,000 epochs (here: id 7)
python3 -c "import sys; sys.path.insert(0,'hpo'); import experiments as e; e.set_experiments_bt_pretrain_long(7)"
./k8s/kube_launcher.sh bt_pretrain ./hpo/bt_pretrain_long.json 7 7

# 4. Fine-tune encoder + decoder end to end, starting from that checkpoint
python3 -c "import sys; sys.path.insert(0,'hpo'); import experiments as e; e.set_experiments_bt_train(7)"
./k8s/kube_launcher.sh bt_train ./hpo/bt_train.json 0 39
```

Outputs go to `/model_pvc/<run>` and `/log_pvc/<run>`, where `<run>` is the params file name. Prefix any launch with `DRY_RUN=1` to print the manifests without submitting.
