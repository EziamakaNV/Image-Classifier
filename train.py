import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

def train_model(data_directory, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print(torch.__version__)
    print(torch.cuda.is_available())
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    print(data_directory + '/train')
    train_data = datasets.ImageFolder(data_directory + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_directory + '/valid', transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    # Load a pre-trained network
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = 1024
    else:
        raise ValueError("Unsupported architecture. Please choose 'vgg16' or 'densenet121'.")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    running_loss = 0
    print_every = 5
    steps = 0
    # Training loop
    for epoch in range(epochs):
        print("starting epoch" + str(epoch))
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/len(trainloader):.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()


    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir + '/checkpoint.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_directory', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture [available: vgg16, ...]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=100, help='Hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true',help='Use GPU for training')

    args = parser.parse_args()
    train_model(args.data_directory, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
