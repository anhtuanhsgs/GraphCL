cat snemi_train.sh 
python -W ignore main.py --env large_multi_5 --gpu-id 0 1 2 3 4 5 6 7 8 --workers 8 --lbl-agents 1 --valid-gpu 1 \
--num-steps 5 --max-episode-length 5 --reward seg --model AttUNet2 --out-radius 12 28 \
--use-masks --size 320 320 --log-period 10 --features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample 1 --data snemi --in-radius 0.8 --lr 1e-4 \
--fgbg-ratio 0.3 --st-fgbg-ratio 0.7 --mer_w 1.0 --spl_w 1.5 --save-period 50 --minsize 20 \
--multi 2 \
--log-dir logs/Jan2019/ \
--dilate-fac 2 \
 \

