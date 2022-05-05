import torch.nn as nn
import torch
import numpy as np
from dataset import ADE20KDataset
from torch.utils.data import DataLoader
from models import Segmenter, MaskTransformer, WNet
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from wnet_loss import soft_n_cut_loss
from timm import create_model
from timm import optim
from timm.models.vision_transformer import VisionTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def training_function(model_architecture, lr=0.001, batch_size=16, epochs=100,
                      save_model=True, save_loss_plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### CREATE DATALOADERS ###
    train_data = ADE20KDataset(split="training")
    test_data = ADE20KDataset(split="validation")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    ### MODEL PARAMETERS ###
    model = model_architecture
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.create_optimizer_v2(model, opt="sgd", lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    loss_func = nn.CrossEntropyLoss(ignore_index=-1) # Ignore Misc Class

    avg_train_loss = []
    avg_test_loss = []
    best_test_loss = np.inf

    for epoch in range(1, epochs + 1):
        print("****** EPOCH: [{}/{}] LR: {} ******".format(epoch, epochs, round(optimizer.param_groups[0]['lr'], 4)))
        train_loss = []
        test_loss = []

        ### Model Training ###
        model.train()
        loop_train = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
        for idx, (img, trgt) in enumerate(loop_train):
            img, trgt = img.to(device), trgt.to(device)
            optimizer.zero_grad()
            forward_out = model(img)

            train_loss_val = loss_func(forward_out, trgt)
            train_loss_val.backward()
            optimizer.step()
            train_loss.append(train_loss_val.item())

            if idx == len(train_dataloader) - 1:
                loop_train.set_description(f"Training")
                loop_train.set_postfix(avg_train_loss=np.array(train_loss).mean())
            else:
                loop_train.set_description(f"Training")
                loop_train.set_postfix(train_loss=train_loss_val.item())

        ### MODEL EVALUATION ###
        model.eval()
        loop_test = tqdm(test_dataloader, total=len(test_dataloader), leave=True)
        with torch.no_grad():
            for idx, (img, trgt) in enumerate(loop_test):
                img, trgt = img.to(device), trgt.to(device)
                forward_out = model(img)
                test_loss_val = loss_func(forward_out, trgt)
                test_loss.append(test_loss_val.item())

                if idx == len(test_dataloader) - 1:
                    loop_test.set_description(f"Evaluate")
                    loop_test.set_postfix(avg_test_loss=np.array(test_loss).mean())
                else:
                    loop_test.set_description(f"Evaluate")
                    loop_test.set_postfix(test_loss=test_loss_val.item())


        avg_train, avg_test = np.mean(train_loss), np.mean(test_loss)
        scheduler.step(avg_test)

        if save_model:
            if avg_test < best_test_loss:
                print("Saving Model")
                best_test_loss = avg_test
                torch.save(model, "outputs/model_store/model_store.pt") ### change model name to save as

        avg_train_loss.append(avg_train)
        avg_test_loss.append(avg_test)

        if save_loss_plot:
            plt.figure(figsize=(10,5))
            plt.title("Training and Validation Loss")
            plt.plot(avg_train_loss, label="val")
            plt.plot(avg_test_loss, label="train")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("trainingloss.png")
            plt.clf()

def train_wnet(model, optimizer, recon_loss, trainloader, device,
               valloader=None, epochs=15):
    enc_train_loss_over_epochs = []
    dec_train_loss_over_epochs = []
    enc_val_loss_over_epochs = []
    dec_val_loss_over_epochs = []
    plt.ioff()
    fig = plt.figure()
    for epoch in tqdm(range(epochs), total=epochs):
        # running loss is the **average** loss for each item in the dataset during this epoch
        enc_running_loss = 0.0
        dec_running_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Your code
            # -------------------------
            # move data onto the device
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Optimization
            optimizer.zero_grad()
            enc_out = model(inputs, enc_rec='enc')
            enc_loss =soft_n_cut_loss(enc_out, inputs)
            enc_loss.backward()
            optimizer.step()
            dec_out = model(inputs, enc_rec='dec')
            dec_loss = recon_loss(inputs, dec_out)
            dec_loss.backward()

            optimizer.step()
            enc_running_loss += enc_loss.cpu().data.numpy()
            dec_running_loss += dec_loss.cpu().data.numpy()

        # Average Loss for each item in the batch
        enc_running_loss = enc_running_loss / len(trainloader)
        dec_running_loss = dec_running_loss / len(trainloader)
        # -------------------------

        enc_train_loss_over_epochs.append(enc_running_loss)
        dec_train_loss_over_epochs.append(dec_running_loss)
        # Note: it can be more readable to overwrite the previous line - end="\r"
        print('Epoch: {}, encoding loss: {:.3f}, decoding loss: {:.3f}'.format(epoch + 1, enc_running_loss,
                                                                               dec_running_loss))

        # If you pass in a validation dataloader then compute the validation loss
        if not valloader is None:
            val_enc_running_loss = 0.0
            val_dec_running_loss = 0.0
            with torch.no_grad():
                for data in valloader:
                    # Your code
                    # -------------------------
                    # move data onto the device
                    val_inputs, val_labels = data
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_enc_out = model(val_inputs, enc_rec='enc')
                    val_dec_out = model(val_inputs, enc_rec='dec')

                    # clear out all variables
                    val_enc_loss = soft_n_cut_loss(val_enc_out, val_inputs)
                    val_dec_loss = recon_loss(val_inputs, val_dec_out)
                    # val_enc_loss = val_enc_loss.item()
                    # val_dec_loss = val_dec_loss.item()
                    val_enc_running_loss += val_enc_loss.cpu().data.numpy()
                    val_dec_running_loss += val_dec_loss.cpu().data.numpy()

                # Average Loss for each item in the batch
                val_enc_running_loss = val_enc_running_loss / len(valloader)
                val_dec_running_loss = val_dec_running_loss / len(valloader)
                # -------------------------
            enc_val_loss_over_epochs.append(val_enc_running_loss)
            dec_val_loss_over_epochs.append(val_dec_running_loss)
            print('Epoch: {}, validation encoding loss: {:.3f}, validation decoding loss: {:.3f}'.format(epoch + 1,
                                                                                                         val_enc_running_loss,
                                                                                                         val_dec_running_loss))

    plt.subplot(2, 1, 1)
    plt.ylabel('Encoding Loss')
    plt.plot(np.arange(epochs), enc_train_loss_over_epochs, color='red', label='train')
    if not valloader is None:
        plt.plot(np.arange(epochs), enc_val_loss_over_epochs, color='blue', label='val')
    plt.title('Loss per Epoch')
    plt.xticks(np.arange(epochs, dtype=int))
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylabel('Decoding Loss')
    plt.plot(np.arange(epochs), dec_train_loss_over_epochs, color='red', label='train')
    if not valloader is None:
        plt.plot(np.arange(epochs), dec_val_loss_over_epochs, color='blue', label='val')
    plt.title('Loss per Epoch')
    plt.xticks(np.arange(epochs, dtype=int))
    plt.grid(True)
    plt.legend()

    plt.show()
    return model

if __name__ == "__main__":
    ViT = VisionTransformer(img_size=384, embed_dim=192, drop_rate=0.2,
                            attn_drop_rate=0.2, drop_path_rate=0.2)
    timm_vit = create_model("vit_tiny_patch16_384", pretrained=True)
    ViT.load_state_dict(timm_vit.state_dict())
    segmenter = Segmenter(encoder=ViT,
                          decoder=MaskTransformer(n_cls=150,
                                                  patch_size=16,
                                                  d_encoder=192,
                                                  n_layers=12,
                                                  n_heads=12,
                                                  d_model=192),
                          n_cls=150,
                          patch_size=16)

    training_function(segmenter)