				________

				 README
				________


Table of Contents
_________________

1 Rotation tasks
2 Hand-written tasks
3 PACS
4 Speech task


This folder contains code accompanying the submission: "Efficient Domain
Generalization via Common-Specific Low-Rank Decomposition".  The folders
are organized into
1. rotation: Rotation tasks on MNIST and Fashion-MNIST
2. hw: Hand-written tasks on LipitK and Nepali Character recognition
   datasets
3. pacs: PACS evaluation with ResNet18
4. speech: Speech utterance classification evaluation

To be able to run the experiments, you need the following packages for
Python3.6
1. Tensorflow <= 1.16
2. PyTorch
3. Keras

The data for hand-written and rotation tasks are either provided or
scripted to automatically download, however the PACS and speech dataset
are to be downloaded and configured in order to run the provided code.

The following sections give more specific instructions on how to run the
code.


1 Rotation tasks
================

  CSD: `python main.py --dataset mnist --classifier mos` ERM: `python
  main.py --dataset mnist --classifier simple`


2 Hand-written tasks
====================

  `python hw_train_and_test.py --dataset lipitk --num_train <number of
  train domains>` `python hw_train_and_test.py --dataset nhcd --lr 1e-4`

  Use flags: `--simple` or `--cg` for ERM or CG baselines.


3 PACS
======

  Download and configure the dataset
  [https://domaingeneralization.github.io/] Define target and source
  domains appropriately.

  `python train_csd.py --train_all --min_scale 0.8 --max_scale 1.0
  --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1
  --source photo cartoon sketch --target art_painting --bias_whole_image
  0.9 --image_size 222`


4 Speech task
=============

  You need to download Speech dataset and extract it in to
  `speech_dataset` folder.  It can be downloaded from:
  [http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz]

  `python train.py --training_percentage <num train domains>
  --train_dir=<checkpoints folder> --model mos2 --seed 0 --learning_rate
  2e-3 --how_many_epochs 500 --lmbda 0.5 --num_uids=<2(K)>`
