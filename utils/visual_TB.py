from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    """
        Defines the functions needed to write statistics and the output images to a tensorboard file.
    
    """
    def __init__(self, log_dir):
        """
            log_dir = folder where the tensorboard log should be stored
        """

        self.summary_writer = SummaryWriter(log_dir=log_dir)

    # Write the learning rate to the tensorboard file
    def write_lr(self, optim, globaliter):
        for i, param_group in enumerate(optim.param_groups):
            self.summary_writer.add_scalar('learning_rate/lr_'+ str(i) , param_group['lr'], globaliter)
        self.summary_writer.flush()
    
    # Write statistics in text form to the tensorboard file. Training time or maximum GPU memory usage.
    def write_text(self, value, statType):
        if statType == 'Time':
            self.summary_writer.add_text('Time','Total Training time: ' + value + ' seconds')
        elif statType == 'Memory':
            self.summary_writer.add_text('GPU', 'Maximum GPU usage: ' + value + ' MiB')
        self.summary_writer.flush()
    
    # Write the training loss to the tensorboard file    
    def write_loss_train(self, value, globaliter):
        self.summary_writer.add_scalar('Loss/train', value, globaliter)
        self.summary_writer.flush()

    # Write the validation loss to the tensorboard file    
    def write_loss_validation(self, value, globaliter, if_testtimes=False):
        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''

        self.summary_writer.add_scalar('Loss/validation'+postfix, value, globaliter)
        self.summary_writer.flush()
    
    # Write the output images to the tensorboard file    
    def write_image(self, images, epoch, if_predict=False, if_testtimes=False, includeHeading = False):
        """
            images: Prediction or ground truth in image form
            epoch: Current epoch of the training
            if_predict: Is it a prediction or a ground truth?
            if_testtimes: Is the validation happening on the testtimes only?
            includeHeading: Is the heading channel included in the training?
        """
        
        
        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''
            
        # restructure the data, save volume, speed and heading separately
        if len(images.shape) == 4:
            _, _, row, col = images.shape
            vol_batch = torch.zeros((3, 1 , row, col))
            speed_batch = torch.zeros((3, 1 , row, col))
            if includeHeading:
                head_batch = torch.zeros((3, 1 , row, col))

            # volume
            vol_batch[0] = images[0,0,:,:]
            vol_batch[1] = images[0,3,:,:]
            vol_batch[2] = images[0,6,:,:]
            # speed
            speed_batch[0] = images[0,1,:,:]
            speed_batch[1] = images[0,4,:,:]
            speed_batch[2] = images[0,7,:,:]

            # heading
            if includeHeading:
                head_batch[0] = images[0,2,:,:]
                head_batch[1] = images[0,5,:,:]
                head_batch[2] = images[0,8,:,:]
            
            
            
        else:
            
            _, _, _, row, col = images.shape
            vol_batch = torch.zeros((3, 1 , row, col))
            speed_batch = torch.zeros((3, 1 , row, col))
            if includeHeading:
                head_batch = torch.zeros((3, 1 , row, col))

            # volume
            vol_batch[0] = images[0,0,0,:,:]
            vol_batch[1] = images[0,1,0,:,:]
            vol_batch[2] = images[0,2,0,:,:]
            
            
            # speed
            speed_batch[0] = images[0,0,1,:,:]
            speed_batch[1] = images[0,1,1,:,:]
            speed_batch[2] = images[0,2,1,:,:]
            # heading
            if includeHeading:
                head_batch[0] = images[0,0,2,:,:]
                head_batch[1] = images[0,1,2,:,:]
                head_batch[2] = images[0,2,2,:,:]
            
        # add images to the tensorboard file
        if if_predict:
            
            vol_batch = torchvision.utils.make_grid(vol_batch, normalize=True)
            self.summary_writer.add_image('prediction'+postfix+'/volume', vol_batch, epoch)

            speed_batch = torchvision.utils.make_grid(speed_batch, normalize=True)
            self.summary_writer.add_image('prediction'+postfix+'/speed', speed_batch, epoch)

            if includeHeading:
                head_batch = torchvision.utils.make_grid(head_batch, normalize=True)
                self.summary_writer.add_image('prediction'+postfix+'/heading', head_batch, epoch)

        else:
            vol_batch = torchvision.utils.make_grid(vol_batch, normalize=True)
            self.summary_writer.add_image('ground_truth'+postfix+'/volume', vol_batch, epoch)

            speed_batch = torchvision.utils.make_grid(speed_batch, normalize=True)
            self.summary_writer.add_image('ground_truth'+postfix+'/speed', speed_batch, epoch)

            if includeHeading:
                head_batch = torchvision.utils.make_grid(head_batch, normalize=True)
                self.summary_writer.add_image('ground_truth'+postfix+'/heading', head_batch, epoch)
        
        # Apply changes to tensorboard file
        self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()