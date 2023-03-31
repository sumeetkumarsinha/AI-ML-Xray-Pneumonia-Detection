
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets
from torchvision.datasets import ImageFolder as ImageFolder
import argparse
import os


try:
    import smdebug
except ModuleNotFoundError:
    print("module 'smdebug' is not installed. Probably an inference container")
    
#Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd
    
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True #disable image truncated error


#Create the data loader function  
def create_data_loaders(data, batch_size):
    
    transform = transforms.Compose ([
        #transforms.Grayscale(num_output_channels=1),
        #converts image to 1 color channel
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
    
    data = torchvision.datasets.ImageFolder(root=data, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                            )
    return data_loader
  
    
#Create the training loop function
def train(model, train_loader, valid_loader, criterion, optimizer, device, args):
    #hook.set_mode(smd.modes.TRAIN)
    
    #epochs=10
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':valid_loader}
    loss_counter=0
    
    
    
    for epoch in range(args.epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 100  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                
                
                
                #For hyperparameter tuning we will use 20% of the dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                   break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
    
    
#Create the model using resnet50    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 2)) #there are 2 output classes - pneumonia, normal
    return model
    






def test(model, test_loader, criterion, device, args):
    print("Testing Model on Whole Testing Dataset")
    
    #hook.set_mode(smd.modes.EVAL)
    
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    print(f"Test set: Average loss: {total_loss}")

    
    
    

def main(args):
    
    os.environ['SM_CHANNEL_TRAIN']
    os.environ['SM_CHANNEL_VALID']
    os.environ['SM_CHANNEL_TEST']
    os.environ['SM_MODEL_DIR']
    
    ### Create data loaders for train, validation and test datasets
    
    train_loader = create_data_loaders(os.environ['SM_CHANNEL_TRAIN'], args.batch_size)
    
    valid_loader = create_data_loaders(os.environ['SM_CHANNEL_VALID'], args.batch_size)
    
    test_loader = create_data_loaders(os.environ['SM_CHANNEL_TEST'], args.batch_size)
    
    
    
    #Initialize a model by calling the net function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    model=net()
    model=model.to(device)
    
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}


    #Create loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
   
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, valid_loader, criterion, optimizer, device, args)
        test(model, test_loader, criterion, device, args)
    
    save_model(model, args.model_dir)
    
    
def model_fn(model_dir):
    return model

def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    print(f"Saving the model to path {path}")
    torch.save(model.state_dict(), path)
    
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="dogImage Classifier")
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    
   
    parser.add_argument(
        "--lr", type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ['SM_MODEL_DIR']
    )
        
        
    args=parser.parse_args()
    
    main(args)
