set -x

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
