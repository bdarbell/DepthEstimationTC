#!/usr/bin/env python

# standard imports
import time

# 3rd party imports
import torch

# local imports
from .configs import DEVICE, MODEL, WEIGHTS_OUT_DIR
from .dataloaders import get_dataloaders
from .loss import tc_loss
from .models.depth_estimation import get_midas_model
from .models.opt_flow_estimation import get_raft_model


def train_model(num_epochs: int = 10) -> None:
    """
    Train the model.

    Parameters:
    - num_epochs (int): The number of epochs to train the model. (default: 10)

    Returns:
    - None
    """

    # Get models and transformations
    midas, transform_midas, midas_ref = get_midas_model(model_type=MODEL)
    raft, transform_raft = get_raft_model()

    # Set the optimizer
    optimizer = torch.optim.Adam(midas.parameters(), lr=1e-3)

    # Get the data loaders
    dataloaders = get_dataloaders()

    # Set the model to training mode
    midas.train()

    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        running_loss = 0.0

        # Iterate over data.
        for imgs, _ in dataloaders['train']:
            # Transform images
            imgs = imgs.to("cpu")
            input_batch = torch.empty((0))
            for img in imgs:
                img = torch.permute(img,(1,2,0))
                input_batch = torch.cat((input_batch, transform_midas(img.numpy())), 0)
            
            # Move data to the device
            input_batch = input_batch.to(DEVICE)
            imgs = imgs.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            with torch.set_grad_enabled(True):
                # Get model outputs
                dms = midas(input_batch)
                dms = torch.nn.functional.interpolate(
                    dms.unsqueeze(1),
                    size=imgs.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                # Compute loss
                loss = tc_loss(imgs=imgs, dms=dms[:,None,:,:], imgs_midas=input_batch, raft=raft, transform_raft=transform_raft, midas_ref=midas_ref)
                
                # Backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * imgs.size(0)/2
            torch.save(midas.state_dict(), WEIGHTS_OUT_DIR)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)

        print('Loss: {:.4f}'.format(epoch_loss))

        # Save the model
        torch.save(midas.state_dict(), WEIGHTS_OUT_DIR)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))