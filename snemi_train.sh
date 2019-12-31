cat snemi_train.sh 
python -W ignore main.py --env modified_AttUNet_3 --gpu-id 0 1 2 3 --workers 8 --lbl-agents 0 \
--num-steps 6 --max-episode-length 6 --reward seg --model AttUNet3 --out-radius 16 32 --use-masks \
--size 256 256 --log-period 10 --features 32 64 128 128 256 512 --entropy-alpha 0.05 --downsample 1 \
--data snemi --in-radius 0.85 --lr 1e-4 --fgbg-ratio 0.5 --st-fgbg-ratio 0.8 \
--mer_w 1.0 --spl_w 1.5 --save-period 50 --minsize 20
