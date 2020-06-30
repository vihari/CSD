if false;
then
    # resnet
    for seed in 0 1 2;
    do
        lr=5e-3
        log="logs/csd_K=2_cswt=0.1t_wts=1e-1bias_seed="$seed"_lr="$lr"_run2_tgt="
        export CUDA_VISIBLE_DEVICES="0"
        python3.6 train_csd.py --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source photo cartoon sketch --target art_painting --bias_whole_image 0.9 --image_size 222 --seed $seed -l $lr > $log"a.log" 2>&1 &

        export CUDA_VISIBLE_DEVICES="1"
        python3.6 train_csd.py --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source photo art_painting sketch --target cartoon --bias_whole_image 0.9 --image_size 222 --seed $seed -l $lr > $log"c.log" 2>&1 &

        export CUDA_VISIBLE_DEVICES="2"
        python3.6 train_csd.py --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source cartoon art_painting sketch --target photo --bias_whole_image 0.9 --image_size 222 --seed $seed -l $lr > $log"p.log" 2>&1 &

        export CUDA_VISIBLE_DEVICES="3"
        python3.6 train_csd.py --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source cartoon photo art_painting --target sketch --bias_whole_image 0.9 --image_size 222 --seed $seed -l $lr > $log"s.log" 2>&1 &
    done
fi

if true;
then
    # alexnet
    lr=1e-3
    # defaults
    mn=1
    rhf=0
    vs=0.1
    j=0.
    bw=1

    for s in 0 1 2;
    do
	log="logs_alexnet/run2_csd_K=2_seed="$s"_lr"$lr"_rhf"$rhf"_j"$j"_cswt=0_wts=1e-1bias_tgt="
	export CUDA_VISIBLE_DEVICES="0"
	python train_csd.py --nesterov --batch_size 128 -s $s --n_classes 7 --learning_rate $lr --network caffenet_csd --val_size $vs --folder_name test --train_all --image_size 225 --min_scale $mn --max_scale 1.0 --random_horiz_flip $rhf --jitter $j --tile_random_grayscale 0.1 --source photo cartoon sketch --target art_painting --bias_whole_image $bw > $log"a.log" 2>&1 &

	export CUDA_VISIBLE_DEVICES="1"
	python train_csd.py --nesterov --batch_size 128 -s $s --n_classes 7 --learning_rate $lr --network caffenet_csd --val_size $vs --folder_name test --train_all --image_size 225 --min_scale $mn --max_scale 1.0 --random_horiz_flip $rhf --jitter $j --tile_random_grayscale 0.1 --source art_painting sketch photo --target cartoon --bias_whole_image $bw > $log"c.log" 2>&1 &
    
	export CUDA_VISIBLE_DEVICES="2"
	python train_csd.py --nesterov --batch_size 128 -s $s --n_classes 7 --learning_rate $lr --network caffenet_csd --val_size $vs --folder_name test --train_all --image_size 225 --min_scale $mn --max_scale 1.0 --random_horiz_flip $rhf --jitter $j --tile_random_grayscale 0.1 --source art_painting cartoon sketch --target photo --bias_whole_image $bw > $log"p.log" 2>&1 &
    
	export CUDA_VISIBLE_DEVICES="3"
	python train_csd.py --nesterov --batch_size 128 -s $s --n_classes 7 --learning_rate $lr --network caffenet_csd --val_size $vs --folder_name test --train_all --image_size 225 --min_scale $mn --max_scale 1.0 --random_horiz_flip $rhf --jitter $j --tile_random_grayscale 0.1 --source art_painting cartoon photo --target sketch --bias_whole_image $bw > $log"s.log" 2>&1 &

    done
fi
