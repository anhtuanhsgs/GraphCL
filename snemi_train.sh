cat snemi_train.sh 
python main.py --env branch --gpu-id 0 1 2 3 4 5 6 7 --workers 1 --lbl-agents 0 --num-steps 5 --max-episode-length 5 --reward seg --model AttUNet2 --out-radius 16 --use-masks --size 128 128 --log-period 10 --features 32 64 128 256 512 --entropy-alpha 0.05 --downsample 2 --data snemi --in-radius 0.7 --lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.5 --mer_w 1.0 --spl_w 1.5 --save-period 50 --minsize 15 --branch 