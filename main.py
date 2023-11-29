import os
import pdb
import random
import argparse
import numpy as np

import torch

## Self-defined
from dataloader import create_dataset
from model import Estimator
from trainer import build_trainer

def parse_args():
	parser = argparse.ArgumentParser()

	## Mode
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--test" , action="store_true", help="Load previous checkpoint for testing only.")

	## Training options
	parser.add_argument("--train_epoch", type=int, default=100)
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--bs", type=int, default=256, help="Batch size during training")
	parser.add_argument("--scale", type=int, default=4, help="sr scale")
	parser.add_argument("--gt", type=str, default="kernel", help="supervise mode, kernel or hq")
	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.999)
	parser.add_argument("--n_feats", type=int, default=64, help="conv channel")
	parser.add_argument("--kernel_size", type=int, default=19, help="predict kernel size")
	parser.add_argument("--report_every", type=int, default=1, help="Display results every 1 epochs.")
	parser.add_argument("--save_kernel_img", type=int, default=0, help="plot pred, gt kernel")
	parser.add_argument("--seed", type=int, default=None, help="Whether to fix random seed or not.")
	parser.add_argument("--device", type=str, default="cuda")

	## Paths
	parser.add_argument("--name", type=str, default="/eva_data2/shlu2240/Dataset/celeba_hq")
	parser.add_argument("--root_path", type=str, default="/eva_data2/shlu2240/Dataset/celeba_hq")
	parser.add_argument("--result_path", type=str, default="./result")
	parser.add_argument("--checkpoint_path", type=str, default="./checkpoints")
	parser.add_argument("--pretrain_path", type=str, default="/eva_data2/shlu2240/explicit_kernel_estimation/checkpoints/estimator_kernel.pt")

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()

	## Set device
	device = torch.device(args.device)
	print("\nDevice: {}\n".format(device))

	if device.type != "cuda":
		raise SystemExit("Not using GPU for training!")

	## Set random seed
	if args.seed is not None:
		print("Random seed: {}\n".format(args.seed))
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		np.random.seed(args.seed)
		random.seed(args.seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

	## Build dataset
	train_loader, val_loader, test_loader = create_dataset(args)

	## Build model
	model = Estimator(args)

	## Build trainer & train (or test)
	trainer = build_trainer(args, model, device)
	
	if args.train:
		print("Start training...")
		trainer.train(train_loader, val_loader)

	elif args.test:
		print("Test only...")
		test_loss = trainer.test(test_loader)

		print(f"Test Accuracy: {test_loss}\n")
