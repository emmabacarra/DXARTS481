'''
------------------------------------------------------------------------------------------------------------------------------------------
Active Modules
------------------------------------------------------------------------------------------------------------------------------------------
'''
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

import random
import time
from datetime import timedelta

import os
import inspect
import logging


class Configuration:
    def __init__(self):
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(profile="full")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = time.strftime('%m-%d-%y__%H-%M-%S', time.localtime())

    def initiate_logs(self):
        # ========================== Logger Configuration ==========================
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%y %H:%M:%S')
        os.makedirs(f'./Training Logs/{self.timestamp}', exist_ok=True)

        # file handler
        log_dir = f'./Training Logs/{self.timestamp}'
        file_handler = logging.FileHandler(log_dir+'/textual.log', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self.logger = logger
        self.logger.info(f'Textual logger configured. Logs will be saved to "{log_dir}/textual.log".')

        # ========================== Tensorboard Configuration ==========================
        self.writer_train = SummaryWriter(log_dir=log_dir+'/tensorboard/train')
        self.writer_val = SummaryWriter(log_dir=log_dir+'/tensorboard/validation')
        logger.info(f'Tensorboard writers created. Tensorboard logs will be saved to "{log_dir}/tensorboard".')

    def save_checkpoint(self, epoch, optimizer):
        os.makedirs(f'./Checkpoints/{self.timestamp}', exist_ok=True)
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'batch_trlosses': self.batch_trlosses,
            'global_step': self.global_index
        }
        path = f'./Checkpoints/{self.timestamp}/epoch_{epoch}_model.pth'
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved to '{path}'.")

    def load_checkpoint(self, optimizer, path):
        if os.path.isfile(path):
            self.logger.info(f"Loading checkpoint from '{path}'")

            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # self.batch_trlosses.extend(checkpoint['batch_trlosses'])
            self.global_index = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch'] + 1
            
            self.logger.info('Checkpoint loaded successfully.')
        else:
            raise FileNotFoundError(f"No checkpoint found at '{path}'")


class Experiment(Configuration):
    def __init__(self, model=None, stochastic=False):
        super().__init__() # inherit methods/properties from Configuration
        '''
        Inheritance:
        self.device
        self.timestamp
        self.initiate_logs()
        self.save_checkpoint()
        self.load_checkpoint()
        '''
        self.model = model.to(self.device)
        self.stochastic = stochastic
        
    def resume(self, timestamp=None, epoch=None):
        """
        Resume training from a previous checkpoint. Must be called before calling load_data() and train() methods.

        :param timestamp: The timestamp of the checkpoint to resume from. Format: 'MM-DD-YY__HH-MM-SS'
        :type timestamp: str
        :param epoch: The epoch to resume from. Training loop will start from the next epoch.
        """
        self.resuming = False

        if timestamp is not None and epoch is not None:
            self.resuming = True
            self.timestamp = timestamp
            self.start_epoch = epoch
        
            if timestamp not in os.listdir('./Training Logs'):
                raise FileNotFoundError(f"No training session found with timestamp '{timestamp}'.")
            
            if f'epoch_{epoch}_model.pth' not in os.listdir(f'./Checkpoints/{timestamp}'):
                raise FileNotFoundError(f"No checkpoint found for epoch {epoch} with timestamp '{timestamp}'.")
            
            if epoch < 1:
                raise ValueError("Epoch must be greater than 0.")
            
        elif (timestamp is None) ^ (epoch is None):
            raise ValueError("Both timestamp and epoch must be provided if resuming training.")
        
        self.initiate_logs()

    def load_data(self, dataset, batch_size, split=0.8, seed=None):
        """
        Create train and validation dataloaders from the dataset.

        :param dataset: The dataset to use for training and validation.
        :param batch_size: The batch size to use for training.
        :param split: The percentage (between 0 and 1) of the dataset to use for training. The rest will be used for validation.
        :param seed: Optional: the random seed to use for splitting the dataset. If None, the split will be random.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.split = split
        self.seed = seed

        # 1) load directory with xy coordinates of each crop (for later references) ------------------
        self.logger.info(f"Loading dataset from '{self.dataset.path()}'...")
        
        self.directory = {}
        with open(self.dataset.path()+'_directory.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(' | ')
                self.directory[parts[0]] = (parts[1], parts[2])

        # 2) train-val split -----------------------------------------------------------------------
        self.logger.info(f"Splitting dataset with seed {seed}..." if seed is not None else 'No seed specified - splitting dataset randomly.')
        
        if seed is not None:
            torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        n_train = int(split * len(self.dataset))
        n_val = len(self.dataset) - n_train
        train_subset, val_subset = random_split(self.dataset, [n_train, n_val], generator=generator)
        
        self.logger.info(f"Dataset split into {n_train} training samples and {n_val} validation samples.")

        # 3) create train and validation/test dataloaders ------------------------------------------
        self.logger.info(f"Creating dataloaders with batch size {batch_size}...")
        
        self.trloader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, generator=generator)
        self.valoader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

        self.num_batches = len(self.trloader) # number of batches in the training set
        self.logger.info(f'DataLoaders created successfully.'
                         f'\nTrain loader size: {len(self.trloader.dataset)} ({self.num_batches} batches)'
                         f'\nValidation loader size: {len(self.valoader.dataset)} ({len(self.valoader)} batches)')
    
    def train(self, epochs, loss_function, optimizer, note_checkpoints, note_progress):
        '''
        Train the model for a specified number of epochs. Logs training progress and saves checkpoints.
        If resuming, call resume() method first.

        :param epochs: The total number of epochs to train for.
        :param loss_function: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param note_checkpoints: Checkpoints to save every n epochs.
        :param note_progress: Progress to note every n epochs.
        '''
        self.total_epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.global_index = 0
        
        if self.resuming:
            self.logger.info(f"Resuming training from session {self.timestamp}, epoch {self.start_epoch}.")
            self.load_checkpoint(optimizer, path=f'./Checkpoints/{self.timestamp}/epoch_{self.start_epoch}_model.pth')
        else:
            self.logger.info("No checkpoint provided for resuming. Training will start from scratch.")
            self.start_epoch = 1
        
        self.params = tuple(self.model.params().items())
        self.start_time = time.time()
        self.note('start')
        try:
            for epoch in range(self.start_epoch, self.total_epochs + 1):
                self.epoch = epoch

                # ========================= training losses =========================
                self.model.train()
                randint = random.randint(0, self.num_batches-6) # -1 to account for 0-indexing, the rest to increase distance from final batch
                for self.i, (batch, filenames) in enumerate(self.trloader, 1):
                    self.global_index = (epoch-1) * self.num_batches + (self.i-1)
                    self.batch_start = time.time()

                    batch = batch.view(batch.size(0), 1, 28, 28)
                    batch = batch.to(self.device)

                    optimizer.zero_grad()

                    # debugging -----------------------
                    if 5 <= self.i <= 10: 
                        self.logger.info(f'Plotting input for batch {self.i}')
                        self.writer_train.add_figure(f'Batch {self.i}, Epoch {epoch} - INPUT',
                                                 self.debug_plots(batch,
                                                    title=f'Batch {self.i}, Epoch {epoch} | Batch Size: {len(batch)}',
                                                    filenames=filenames),
                                                 global_step=self.global_index)
                    # ---------------------------------

                    outputs = self.model(batch)
                    self.batch_loss = self.loss_function(batch, outputs)

                    # debugging -----------------------
                    if 5 <= self.i <= 10: 
                        self.logger.info(f'Plotting output for batch {self.i}')
                        self.writer_train.add_figure(f'Batch {self.i}, Epoch {epoch} - OUTPUT',
                                                 self.debug_plots(outputs,
                                                    title=f'Batch {self.i}, Epoch {epoch} | Loss: {self.batch_loss.item():.8f}',
                                                    compare=batch),
                                                 global_step=self.global_index)
                    # ---------------------------------
                    
                    self.note('train')
                    self.batch_loss.backward()
                    optimizer.step()

                # ========================= validation losses =========================
                self.logger.info(f'Calculating validation for epoch {self.epoch}.')
                self.model.eval()
                with torch.no_grad():
                    tot_valoss = 0
                    for batch, filenames in self.valoader:

                        batch = batch.view(batch.size(0), 1, 28, 28)
                        batch = batch.to(self.device)
                        
                        outputs = self.model(batch)
                        batch_loss = self.loss_function(batch, outputs)

                        tot_valoss += batch_loss.item()

                    self.avg_val_loss = tot_valoss / len(self.valoader)
                    self.note('validation')
                
                # ========================= Progress Checkpoints =========================
                if self.epoch % note_checkpoints == 0 or self.epoch == epochs:
                    self.note('checkpoint')
                
                if self.epoch % note_progress == 0 or self.epoch == epochs:
                    self.note('progress')
            # \\\ END OF TRAINING LOOP \\\

            self.note('end')
            self.writer_train.close()
            self.writer_val.close()
            
        except KeyboardInterrupt:
            self.logger.warning("Training was interrupted by the user.")
            self.note('checkpoint')
            self.writer_train.close()
            self.writer_val.close()

        except (Exception, ValueError, TypeError) as e:
            self.logger.error(f"An error has occurred: {e}", exc_info=True)
            self.note('checkpoint')
            self.note('progress')
            self.writer_train.close()
            self.writer_val.close()
            raise
    
    def note(self, mode):
        '''
        For logging training progress and saving checkpoints.
        
        :param mode: The mode of the note. Must be 'start', 'train', 'validation', 'progress', 'checkpoint', or 'end'.
        :type mode: str
        '''
        elapsed_time = timedelta(seconds=time.time() - self.start_time)
        formatted_time = str(elapsed_time).split(".")[0] + f".{int(elapsed_time.microseconds / 10000):02d}"

        if mode == 'start':
            self.logger.info(f'Training initiated with the following parameters:'
                            f'\nModel Parameters: {self.params} | Stochastic = {self.stochastic}'
                            f'\nStart Epoch: {self.start_epoch} | Total Epochs: {self.total_epochs}'
                            f'\nBatch Size: {self.batch_size} | Training Data Size: {len(self.trloader.dataset)} | Validation Data Size: {len(self.valoader.dataset)}'
                            f'\nOptimizer: {self.optimizer}'
                            f'\nLoss Function: {inspect.getsource(self.loss_function)}')
            self.logger.info(f'Image Data Size: {self.trloader.dataset[0][0].shape}')

        elif mode == 'train':
            batch_time = time.time() - self.batch_start
            learning_rate = self.optimizer.param_groups[0]['lr']

            self.writer_train.add_scalar('loss', self.batch_loss.item(), self.global_index)
            # batch_log = f'({formatted_time}) | [{self.epoch}/{epochs}] Batch {i} ({batch_time:.3f}s) | LR: {learning_rate} | KLW: {klw}, KLD (loss): {kld:.3f}, Rec. Loss: {reconstruction_loss:.8f} | Total Loss: {batch_loss.item():.8f}'
            batch_log = f'({formatted_time}) | [{self.epoch}/{self.total_epochs}] Batch {self.i} ({batch_time:.3f}s) | LR: {learning_rate} | Loss: {self.batch_loss.item():.8f}'
            self.logger.info(batch_log)
        
        elif mode == 'validation':
            learning_rate = self.optimizer.param_groups[0]['lr']

            self.writer_val.add_scalar('loss', self.avg_val_loss, self.global_index)
            # val_log = f'({formatted_time}) | VALIDATION (Epoch {self.epoch}/{epochs}) | LR: {learning_rate} | KLW: {klw}, KLD (loss): {kld:.3f}, Rec. Loss: {reconstruction_loss:.8f} | Total Loss: {avg_val_loss:.8f} -----------'
            val_log = f'({formatted_time}) | VALIDATION (Epoch {self.epoch}/{self.total_epochs}) | Gl Idx: {self.global_index} | LR: {learning_rate} | Loss: {self.avg_val_loss:.8f} -----------'
            self.logger.info(val_log)
        
        elif mode == 'progress':
            self.evaluate()
            self.writer_val.add_figure(f'Generated Samples, Epoch {self.epoch}', self.generate_samples(num_images=50), global_step=self.global_index)
            self.logger.info(f'Generated images created for epoch {self.epoch}.')

            # self.logger.info(f'Creating latent space plot for epoch {self.epoch}...')
            # latent_vectors = self.latent_space()
            # self.logger.info(f'Latent vectors shape: {latent_vectors.shape}')
            # self.writer_val.add_embedding(latent_vectors,  tag=f'Latent Space, Epoch {self.epoch}', global_step=self.global_index)
            # self.logger.info(f'Latent space plot created for epoch {self.epoch}.')
        
        elif mode == 'checkpoint':
            self.save_checkpoint(self.epoch, self.optimizer)

        elif mode == 'end':            
            self.logger.info(
                '\n===========================================================================================\n'
                f'----  TRAINING SUMMARY FOR SESSION {self.timestamp}  ----\n'
                f'\nDataset: {self.dataset.path()} | Training Split: {self.split} | Seed: {self.seed}'
                f'\nModel Parameters: {self.params} | Stochastic = {self.stochastic}'
                f'\nCompleted Epochs: {self.epoch}/{self.total_epochs} | Total Training Time: {formatted_time}'
            )

        else:
            raise ValueError("Invalid mode.")
    
    def evaluate(self, threshold=0.01):
        '''
        Calculate the accuracy of the model on the validation set.

        :param threshold: The threshold to include a pixel value as correctly reconstructed.
        '''
        self.model.eval()
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for images, filenames in self.valoader:
                images = images.to(self.device)
                images = images.view(images.size(0), 1, 28, 28)

                if self.stochastic:
                    reconstruction_images, _, _ = self.model(images)
                if not self.stochastic:
                    reconstruction_images = self.model(images)
                reconstruction_images = reconstruction_images.view_as(images)

                # Calculate the number of correctly reconstructed pixels
                correct_pixels = (torch.abs(images - reconstruction_images) < threshold).type(torch.float).sum().item()
                total_correct += correct_pixels
                total_pixels += images.numel()
        
        self.accuracy = total_correct / total_pixels
        self.logger.info(f'Calculated Accuracy: {self.accuracy:.3f} | Total Correct Pixels: {total_correct}/{total_pixels} | Value Threshold: {threshold}')


    # plot generated samples
    def generate_samples(self, num_images=10, filesave=False):
        from scipy.stats import entropy

        self.model.eval()
        with torch.no_grad():
            data_iter = iter(self.valoader)

            images, filenames = next(data_iter)
            images = images[:num_images].to(self.device)
            filenames = filenames[:num_images]
            images = images.view(images.size(0), 1, 28, 28)

            if self.stochastic:
                reconstruction_images, _, _ = self.model(images)
            if not self.stochastic:
                reconstruction_images = self.model(images)
            reconstruction_images = reconstruction_images.cpu()

        cols = min(num_images, 5)
        rows = (num_images + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, 4 * rows))
        gridspec = fig.add_gridspec(nrows=rows, ncols=11)

        # Create subplots for original and reconstructed images with distinct background colors
        axes_original = fig.add_subplot(gridspec[:, 0:6], facecolor='lightblue')
        axes_reconstructed = fig.add_subplot(gridspec[:, 6:11])

        # Turn off the axes for the overall subplots
        axes_original.axis('off')
        axes_reconstructed.axis('off')


        for i in range(num_images):
            row = i // cols
            col = i % cols

            filename = filenames[i]

            img_in = torch.squeeze(images[i]).cpu().numpy()
            img_out = torch.squeeze(reconstruction_images[i]).cpu().numpy()
            # print("img_in, img_out shapes:", img_in.shape, img_out.shape)

            # shannon entropy
            hist, _ = np.histogram(img_in.flatten(), bins=100)
            entropy_in = entropy(hist)

            # pearson correlation coefficient and p-value
            pcc, pval = pearsonr(img_in.flatten(), img_out.flatten())

            # structural similarity index
            ssim = structural_similarity(img_in, img_out, win_size=11, data_range=img_out.max() - img_out.min())

            # mean squared error
            # mse = mean_squared_error(img_in, img_out)
            mse = self.loss_function(img_in, img_out).item()


            # original image before forward pass
            ax1 = fig.add_subplot(gridspec[row, col], facecolor='lightblue')
            ax1.imshow(img_in, cmap='gray', filternorm=False)
            ax1.set_title(f"{filename}", fontsize=10)
            ax1.set_xlabel(f'Dimensions: {img_in.shape}'
                           f'\nLocation: {self.directory[filename]}' 
                           f'\nMin: {img_in.min():.3f}' 
                           f'\nMax: {img_in.max():.3f}'
                           f'\nEntropy: {entropy_in:.3f}', fontsize=7)
            ax1.set_xticks([]), ax1.set_yticks([])

            # reconstructed image after forward pass
            ax2 = fig.add_subplot(gridspec[row, col + 6])
            ax2.imshow(img_out, cmap='gray', filternorm=False)
            ax2.set_title(f"{filename}", fontsize=10)
            ax2.set_xlabel(f'Dimensions: {img_out.shape}'
                           f'\nMin: {img_out.min():.3f}' 
                           f'\nMax: {img_out.max():.3f}'
                           f'\nPCC: {pcc:.3f} | P-Value: {pval:.3f}' 
                           f'\nMSE: {mse:.3f} | SSIM: {ssim:.3e}', fontsize=7)
            ax2.set_xticks([]), ax2.set_yticks([])

        # Set overall titles for each half
        axes_original.set_title("Original Images", weight='bold', fontsize=15, pad=20)
        axes_reconstructed.set_title("Reconstructed Images", weight='bold', fontsize=15, pad=20)

        # Set the overall title for the entire figure
        fig.suptitle(f"Accuracy: {self.accuracy:.3f}", fontsize=20, fontweight='bold', y=1)

        plt.tight_layout(pad=3)
        if filesave == False:
            return fig
        elif filesave == True:
            os.makedirs('./Generated Samples', exist_ok=True)
            plt.savefig(f"./Generated Samples/{self.timestamp}.png")
        elif isinstance(filesave, str):
            plt.savefig(filesave)
        return fig
    
    def debug_plots(self, batch, title, compare=None, filenames=None):
        fig, ax = plt.subplots(figsize=(15, 15))

        # batch shape should be: torch.Size([100, 1, 28, 28])
        img_grid = make_grid(batch, nrow=10, normalize=True, padding=2, scale_each=True) 
        # normalizing for visualization shouldn't affect training
        # needs to also have scale_each=True to normalize each image separately
        # normalize and scale_each need to be True for image to be displayed correctly
        # img_grid shape: torch.Size([3, 302, 302]) <-- dimensions of the grid picture itself
        ax.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), cmap='grey')
        ax.axis("off")

        batch_size, channels, height, width = batch.shape
        # ax.set_title(f"Batch Dimensions: {batch.shape}", fontsize=16)
        ax.set_title(title, fontsize=16, weight='bold')

        # if isinstance(batch, torch.Tensor):
        #     batch = batch.cpu().detach()
        if not isinstance(batch, torch.Tensor):
            batch = batch.cpu()
        for i in range(10):
            for j in range(10):
                idx = i * 10 + j
                img = batch[idx]

                height, width = img.shape[1], img.shape[2]
                min_val, max_val = img.min().item(), img.max().item()

                # image captions
                x_pos = j * (width + 2) + width // 2  # x-position of the text
                y_pos = i * (height + 2) + height + 2    # y-position of the text

                if filenames is not None:
                    ax.text(x_pos, y_pos,
                        f"Location: {self.directory[filenames[idx]]}"
                        f"\nmin:{min_val:.2f} max:{max_val:.2f}",
                        fontsize=8, ha='center', va='top', color='white', backgroundcolor='black'
                    )
                if compare is not None: # intended for when batch=output, compare=batch
                    compare_img = compare[idx]
                    # mse = mean_squared_error(compare_img.cpu().numpy(), img.numpy())
                    mse = self.loss_function(compare_img, img).item()

                    ax.text(x_pos, y_pos,
                    f"MSE: {mse:.3f}"
                    f"\nmin:{min_val:.2f} max:{max_val:.2f}",
                    fontsize=8, ha='center', va='top', color='white', backgroundcolor='black'
                )
        plt.tight_layout()

        # fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))  # Set the figure size for the histogram
        # # dimensions: batch number, channel, height, width
        # pixel_values = batch.flatten().numpy()

        # axs2[0].hist(pixel_values, bins=200, alpha=0.7, log=True)
        # axs2[0].set_title('Pixel Value Histogram (Logarithmic Scale)', fontsize=15, fontweight='bold')
        # axs2[0].set_xlabel('value')
        # axs2[0].set_ylabel('count')

        # axs2[1].hist(pixel_values, bins=200, alpha=0.7, log=False)
        # axs2[1].set_title('Pixel Value Histogram (Linear Scale)', fontsize=15, fontweight='bold')
        # axs2[1].set_xlabel('value')
        # axs2[1].set_ylabel('count')

        return fig

    # plot latent space
    def latent_space(self):
        latent_vectors = []
        self.model.eval()
        with torch.no_grad():
            for batches, filenames in self.valoader:
                batches = batches.view(batches.size(0), 1, 28, 28).to(self.device)
                
                if self.stochastic:
                    h = self.model.encoder(batches)
                    z, mu, logvar = self.model.bottleneck(h)
                    latent_vectors.append(mu.cpu().numpy())
                '''
                need to fix this for deterministic models
                '''
                if not self.stochastic:
                    h = self.model.encoder(batches)
                    z = self.model.fc1(h)
                    latent_vectors.append(z.cpu().numpy())
            
        latent_vectors = np.concatenate(latent_vectors, axis=0)

        return latent_vectors
    
    # plot reconstructions in latent space
    def prec(self, rangex=(-5, 10), rangey=(-10, 5), n=12, latent_dims = (0, 1)):
        '''
        range in the latent space to generate:
            rangex = range of x values
            rangey = range of y values

        n = number of images to plot
        '''
        w = self.valoader.dataset[0][0].shape[0]  # image width
        img = np.zeros((n*w, n*w))
        for i, y in enumerate(np.linspace(*rangey, n)):
            for j, x in enumerate(np.linspace(*rangex, n)):
                if self.model.latent_dim > 2:
                    # Initialize a latent vector with zeros
                    z = torch.zeros((1, self.model.latent_dim)).to(self.device)
                    # Set the chosen dimensions to the corresponding x, y values
                    z[0, latent_dims[0]] = x
                    z[0, latent_dims[1]] = y
                    # Project other dimensions onto this plane with random values
                    remaining_dims = [dim for dim in range(self.model.latent_dim) if dim not in latent_dims]
                    z[0, remaining_dims] = torch.randn(len(remaining_dims)).to(self.device)
                else:
                    z = torch.Tensor([[x, y]]).to(self.device)
                
                x_hat = self.model.decoder(z)
                x_hat = x_hat.to('cpu').detach().numpy()
                # Convert to single channel if necessary (from when using png with rgb channels)
                if x_hat.shape[0] == 3:
                    x_hat = np.dot(x_hat.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140])
                elif x_hat.shape[0] == 1:
                    x_hat = x_hat.squeeze(0)  # Remove the channel dimension if it's a single channel
                else:
                    continue
                
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat

        plt.title(f"Latent Space Images", fontsize = 15, fontweight = 'bold')
        plt.xlabel(f"Dimension {latent_dims[0]}", fontsize = 12)
        plt.ylabel(f"Dimension {latent_dims[1]}", fontsize = 12)
        plt.imshow(img, extent=[*rangex, *rangey])
        plt.tight_layout()
        os.makedirs('./Latent Space Plots', exist_ok=True)
        plt.savefig(f"./Latent Space Plots/{self.timestamp}.png")
