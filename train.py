import copy
import time

import torch
from torchvision.models import resnet18
import torch.utils.data
import torchvision.datasets
from torchvision import transforms
from tqdm import tqdm

import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = {}


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    B, _C, H, W = inputs.shape
                    assert _C == 1
                    assert outputs.shape == (B, 10)
                    assert labels.shape == (B,)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            wandb.log(
                {"epoch": epoch, phase + "/loss": epoch_loss, phase + "/acc": epoch_acc}
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


normalize = transforms.Normalize([72.9404 / 255], [90.0212 / 255])
data_transforms = {
    "train": transforms.Compose(
        [
            # transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(lambda x: x.to(device)),
        ]
    ),
    "val": transforms.Compose(
        [
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(lambda x: x.to(device)),
        ]
    ),
}

train_data = torchvision.datasets.FashionMNIST(
    "fashion-mnist",
    train=True,
    download=True,
    transform=data_transforms["train"],
)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
)

test_data = torchvision.datasets.FashionMNIST(
    "fashion-mnist", train=False, transform=data_transforms["val"]
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=64,
    shuffle=True,
)

model = resnet18()
model.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

if __name__ == "__main__":
    wandb.init(config=args, save_code=True)
    try:
        model = train_model(
            model, {"train": train_loader, "val": test_loader}, criterion, optimizer
        )
        torch.save(model.state_dict(), "weights.pkl")
    finally:
        wandb.finish()
