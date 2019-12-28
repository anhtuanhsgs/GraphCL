cat cremi_train.sh 
python main.py --env branch --gpu-id 0 1 2 3 4 5 6 7 --workers 7 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg --model AttUNet2 \
--out-radius 4 8 --use-masks --size 160 160 --log-period 10 --features 16 32 64 128 256 512 \
--entropy-alpha 0.05 --downsample 4 --data cremi --in-radius 0.85 --lr 8e-5 --fgbg-ratio 0.4 --st-fgbg-ratio 0.6 \
--mer_w 1.0 --spl_w 1.8 --save-period 50 --minsize 12 
