cat snemi_train.sh 
python -W ignore main.py --env ins_mer_0.4_spl_0.6 --gpu-id 3 4 5 6 7 --workers 14 --lbl-agents 7 --valid-gpu 3 \
--num-steps 5 --max-episode-length 5 --reward seg --model AttUNet2 --out-radius 8 \
--use-masks --size 256 256 --log-period 10 --features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample 1 --data snemi --in-radius 1.0 --lr 1e-4 \
--fgbg-ratio 0.7 --st-fgbg-ratio 0.7 --mer_w 0.4 --spl_w 0.6  --save-period 50 --minsize 20 \
--max-temp-steps 99 \
--multi 1 \
--T0 0 \
--log-dir logs/Fer2019/ \
--dilate-fac 2 \
 \

