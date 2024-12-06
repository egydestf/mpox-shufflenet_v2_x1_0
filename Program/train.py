import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from Utils.getData import getImageLabel


def main():
    BATCH_SIZE = 32
    EPOCH = 25
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    folds = [1, 2, 3, 4, 5]

    aug_train_loader = DataLoader(getImageLabel(augment=f'./dataset/Augmented Images/Augmented Images/FOLDS_AUG/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    origin_train_loader = DataLoader(getImageLabel(origin=f'./dataset/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(getImageLabel(origin=f'./dataset/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Valid']), batch_size=BATCH_SIZE, shuffle=True)
    
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    model = shufflenet_v2_x1_0(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 6)
    )
    
    device = torch.device('cpu')
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_valid_all = [], []
    for epoch in range(EPOCH):
        train_loss = 0
        valid_loss = 0
        model.train()
        
        # augmented
        for batch, (src, trg) in enumerate(aug_train_loader):
            src, trg = src.to(device), trg.to(device)
            src = torch.permute(src, (0, 3, 1, 2))
            
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # original
        for batch, (src, trg) in enumerate(origin_train_loader):
            src, trg = src.to(device), trg.to(device)
            src = torch.permute(src, (0, 3, 1, 2))
            
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation
        model.eval()
        with torch.no_grad():
            for batch, (src, trg) in enumerate(valid_loader):
                src, trg = src.to(device), trg.to(device)
                src = torch.permute(src, (0, 3, 1, 2))

                pred = model(src)
                loss = loss_function(pred, trg)
                valid_loss += loss.detach().cpu().numpy()

        loss_train_all.append(train_loss / (len(aug_train_loader) + len(origin_train_loader)))
        loss_valid_all.append(valid_loss / len(valid_loader))
        print("epoch = ", epoch + 1, "train loss = ", train_loss / (len(aug_train_loader) + len(origin_train_loader)), ", validation loss = ", valid_loss / len(valid_loader))

        # Save model checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / (len(aug_train_loader) + len(origin_train_loader)),
            }, "./ShuffleNetV2_" + str(epoch + 1) + ".pt")

    # EPOCH Plot
    plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
    plt.plot(range(EPOCH), loss_valid_all, color="#3399e6", label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./training.png")

if __name__ == "__main__":
    main()