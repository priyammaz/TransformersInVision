import torch.nn as nn
import torch
import numpy as np
from dataset import ADE20KDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


def training_function(model_architecture, custom_loss=None, lr=0.0005, batch_size=32, epochs=100,
                      save_model=True, save_loss_plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### CREATE DATALOADERS ###
    train_data = ADE20KDataset(split="training")
    test_data = ADE20KDataset(split="validation")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    ### MODEL PARAMETERS ###
    model = model_architecture.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    loss_func = nn.CrossEntropyLoss()

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

        if save_model:
            if avg_test < best_test_loss:
                print("Saving Model")
                best_test_loss = avg_test
                torch.save(model, "model_store.pt") ### change model name to save as

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







