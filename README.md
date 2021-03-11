# codes.lstp.rec
## Overview
This repo contains the LSTPR model and the seven baselines. Also, the Amazon-beauty dataset is included.
```
├── data (directory for original graph files, LSTP graph files, and field files)
├── emb (directory for embedding files)
├── preprocess.py (generate LSTP graph files)
├── field.py  (generate field files for HOP-Rec and LSTPR)
├── lfm-bpr.py (baseline: BPR)
├── lfm-warp.py (baseline: WARP)
├── smore (baseline: HOP-Rec)
├── SkewOPT (baseline: SkewOPT)
├── LightGCN (baseline: LightGCN)
├── Caser (baseline: Caser)
├── CosRec (baseline: CosRec)
├── predict.py (evaluation code for BPR, WARP, HOP-Rec, SkewOPT, and HOP-Rec)
├── utils.py (contains evaluation metrics)
├── run.sh (script file for preprocessing and all methods' usages)
```
## Abstract
Considering the temporal order of user-item interactions for recommendation forms a novel class of recommendation algorithms in recent years, among which sequential recommendation models are the most popular approaches. Although, theoretically, such fine-grained modeling should be beneficial to the recommendation performance, these sequential models in practice greatly suffer from the issue of data sparsity as there are a huge number of combinations for item sequences. To address the issue, we propose LSTPR, a graph-based matrix factorization model that incorporates both high-order graph information and long short-term user preferences into the modeling process. LSTPR explicitly distinguishes long-term and short-term user preferences and enriches the sparse interactions via random surfing on the user-item graph. Experiments on three recommendation datasets with temporal user-item information demonstrate that the proposed LSTPR model achieves significantly better performance than the seven baseline methods.

## Usages
All usages are written in the ```run.sh``` file. Use ```bash run.sh``` to run the script.

For parameter tuning and requirements, please follow the references. Note that the default parameters are same as those in the LSTPR paper.
- LightFM: https://github.com/lyst/lightfm
- SMORe: https://github.com/cnclabs/smore
- LightGCN: https://github.com/gusye1234/LightGCN-PyTorch
- Caser: https://github.com/graytowne/caser_pytorch
- CosRec: https://github.com/zzxslp/CosRec

```
#Generate the LSTP graph file and the field file for LSTPR
python3 preprocess.py
python3 field.py

#BPR
python3 lfm-bpr.py --train ./data/beauty_train.txt --save ./emb/beauty_lfm-bpr.txt --dim 100 --worker 16
python3 predict.py --emb_file ./emb/beauty_lfm-bpr.txt --dataset beauty --K 10
python3 predict.py --emb_file ./emb/beauty_lfm-bpr.txt --dataset beauty --K 20

#WARP
python3 lfm-warp.py --train ./data/beauty_train.txt --save ./emb/beauty_lfm-warp.txt --dim 100 --worker 16
python3 predict.py --emb_file ./emb/beauty_lfm-warp.txt --dataset beauty --K 10
python3 predict.py --emb_file ./emb/beauty_lfm-warp.txt --dataset beauty --K 20

#HOP-Rec
./smore/cli/hoprec -train ./data/beauty_train.txt -field ./data/beauty_field.txt -save ./emb/beauty_hoprec.txt -dimensions 100 -threads 16 -sample_times 200
python3 predict.py --emb_file ./emb/beauty_hoprec.txt --dataset beauty --K 10
python3 predict.py --emb_file ./emb/beauty_hoprec.txt --dataset beauty --K 20

#Skew-OPT
./SkewOPT/cli/SkewOPT -train ./data/beauty_train.txt -save ./emb/beauty_skewopt.txt -dimensions 100 -threads 16 -sample_times 200
python3 predict.py --emb_file ./emb/beauty_skewopt.txt --dataset beauty --K 10
python3 predict.py --emb_file ./emb/beauty_skewopt.txt --dataset beauty --K 20

#LightGCN
cd LightGCN/code/ && python3 main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="beauty" --topks="[10,20]" --recdim=100

#Caser
cd Caser/ && python3 train_caser.py

#CosRec
cd CosRec/ && python3 train.py --dataset=beauty --d=100 --fc_dim=150 --l2=1e-6 --n_iter=2000 --learning_rate=1e-4

#LSTPR
./smore/cli/hoprec -train ./data/beauty_lstp_5.txt -field ./data/beauty_lstp_field.txt -save ./emb/beauty_lstpr.txt -dimensions 100 -threads 16 -sample_times 200
python3 predict.py --emb_file ./emb/beauty_lstpr.txt --dataset beauty --K 10
python3 predict.py --emb_file ./emb/beauty_lstpr.txt --dataset beauty --K 20
```
