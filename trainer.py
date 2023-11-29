import pdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def convolution2d(input_tensor, custom_kernel, stride, padding=0):
    # input_tensor : [batch_size, c, image_size, image_size]
    # custom_kernel : [batch_size, 1, kernel_size, kernel_size]
    # stride : convolution stride 
    # perform convolution on every channel with custom kernel
    N, C, H, W = input_tensor.shape
    output_channels = []

    # Loop through each input channel and apply the custom convolution
    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].unsqueeze(1)  # Select the channel
        input_channel = input_channel.view(1, N, H, W)
        output_channel = F.conv2d(input_channel, custom_kernel, stride=stride, padding=padding, groups=N)
        output_channel = output_channel.view(N, 1, output_channel.shape[2], output_channel.shape[3])
        output_channels.append(output_channel)

    # Stack the output channels along the channel dimension
    return torch.cat(output_channels, dim=1)

def build_trainer(args, model, device):
    print("Build trainer...")
    trainer = Trainer(args, model, device)
    return trainer

def data_transform(X):
        return 2 * X - 1.0

class Trainer:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device

        self.model.to(self.device)

        ## Load model checkpoint if test only
        if self.args.test:
            model_path = self.args.pretrain_path

            print(f"Loading model checkpoint from {model_path}\n")
            self.model.load_state_dict(
                torch.load(model_path)
            )


    def train(self, train_loader, val_loader):
        """Train the model"""
        best = {"test_loss": 10**10}

        loss_fct = nn.MSELoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=[self.args.beta1, self.args.beta2])

        for epoch in tqdm(range(self.args.train_epoch)):
            total_loss = 0
			
            self.model.train()
            for hq, inputs, kernels in train_loader:
                ## Move to device (GPU)
                inputs = inputs.to(self.device)
                inputs = data_transform(inputs)
                kernels = kernels.to(self.device)

                ## Forward
                self.model.zero_grad()
                pred_kernels = self.model(inputs)
                if self.args.gt == "kernel":
                    pred_flat = pred_kernels.view(pred_kernels.shape[0], 1, -1)
                    gt_flat = kernels.view(kernels.shape[0], 1, -1)
                    loss = loss_fct(pred_flat, gt_flat)
                elif self.args.gt == "hq":
                    hq = hq.to(self.device)
                    pred_lq = convolution2d(hq, pred_kernels, stride=self.args.scale, padding=self.args.kernel_size // 2)
                    pred_lq = pred_lq.view(pred_lq.shape[0], pred_lq.shape[1], -1)
                    gt_lq = inputs.view(inputs.shape[0], inputs.shape[1], -1)

                    loss = loss_fct(pred_lq, gt_lq)  

                ## Backward
                loss.backward()
                optimizer.step()

                ## Update Statistics
                total_loss += loss.item()

            test_loss = self.test(val_loader)
            if test_loss < best['test_loss']:
                print('saving model')
                best['test_loss'] = test_loss
                model_path = os.path.join(self.args.checkpoint_path, f'estimator_{self.args.gt}.pt')
                torch.save(self.model.state_dict(), model_path)

            ## Display statistics
            if epoch % self.args.report_every == 0:
                print(
                    f"Epoch: {epoch:2d}/{self.args.train_epoch:2d}, Train Loss: {total_loss / len(train_loader)}, Val loss: {test_loss}"
                )
            
    
    def test(self, test_loader):
        """Test the model every epoch during training."""
        all_labels, all_preds = [], []
        loss_fct = nn.MSELoss()
        total_loss = 0
        self.model.eval()
        with torch.no_grad(): ## Important: or gpu will go out of memory
            for idx, (hq, inputs, kernels) in enumerate(tqdm(test_loader)):
                ## Move to device (GPU)
                inputs = inputs.to(self.device)     
                kernels = kernels.to(self.device)
                inputs = data_transform(inputs)
                pred_kernels = self.model(inputs)

                if self.args.save_kernel_img:
                    self.plot_kernel(kernels, pred_kernels, idx)

                if self.args.gt == "kernel":
                    pred_flat = pred_kernels.view(pred_kernels.shape[0], 1, -1)
                    gt_flat = kernels.view(kernels.shape[0], 1, -1)
                    loss = loss_fct(pred_flat, gt_flat)
                elif self.args.gt == "hq":
                    hq = hq.to(self.device)
                    pred_lq = convolution2d(hq, pred_kernels, stride=self.args.scale, padding=self.args.kernel_size // 2)
                    pred_lq = pred_lq.view(pred_lq.shape[0], pred_lq.shape[1], -1)
                    gt_lq = inputs.view(inputs.shape[0], inputs.shape[1], -1)
                    loss = loss_fct(pred_lq, gt_lq)  

                total_loss += loss.item()

            

        return total_loss / len(test_loader)
    
    def plot_kernel(self, gt_kernel, pred_kernel, batch_idx):
        batch_size, _, kernel_size, _ = pred_kernel.shape

        pred_kernels_np = pred_kernel.detach().cpu().numpy()
        gt_kernels_np = gt_kernel.detach().cpu().numpy()
        os.makedirs(f'./{self.args.result_path}/{self.args.gt}_supervise/kernel_image/', exist_ok=True)
        for i in range(batch_size):
            pred_kernel = pred_kernels_np[i, 0, :, :]
            gt_kernel = gt_kernels_np[i, 0, :, :]

            # Plotting
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(pred_kernel, cmap='viridis', interpolation='nearest')
            plt.title('Prediction Kernel')

            plt.subplot(1, 2, 2)
            plt.imshow(gt_kernel, cmap='viridis', interpolation='nearest')
            plt.title('Ground Truth Kernel')

            plt.tight_layout()
            plt.savefig(f'./{self.args.result_path}/{self.args.gt}_supervise/kernel_image/{batch_idx}_{i}.png')
            plt.close()



