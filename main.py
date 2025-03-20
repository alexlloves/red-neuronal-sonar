from SonarDataset import SonarDataset
from torch.utils.data import random_split
import torch
from Model import *
import torch.nn as nn

def split_dataset(dataset):
    lonxitudeDataset = len(dataset)
    tamTrain = int(lonxitudeDataset * 0.8)
    tamVal = lonxitudeDataset - tamTrain
    
    print(f"Tam dataset: {lonxitudeDataset} train: {tamTrain} tamVal: {tamVal}")
    
    train_set, val_set = random_split(dataset, [tamTrain, tamVal])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2,
                                               shuffle=True, drop_last=False)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                                    shuffle=False, num_workers=2)

    return train_loader, validation_loader

dataset = SonarDataset("data/sonar.all-data",".")
print(dataset[0])
train_loader, validation_loader = split_dataset(dataset)



model = Model(60)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
print(model)

entradaProba,dest = next(iter(train_loader))
print("Entrada:")
print(entradaProba)
print("Desexada:")
print(dest)
saida = model(entradaProba)
print("Saída:")
print(saida)
loss_fn(saida, dest)


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

EPOCHS = 100
loss_list = torch.zeros((EPOCHS,))
accuracy_list = torch.zeros((EPOCHS,))
lonxitudeDataset = len(dataset)

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))


    model.train(True)
    avg_loss = train_one_epoch(epoch)
    loss_list[epoch] = avg_loss
    model.train(False)
    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)

        correct = (torch.argmax(voutputs, dim=0) == vlabels).type(torch.FloatTensor)
        accuracy_list[epoch] += correct.sum()
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {} {}/{}'.format(avg_loss, avg_vloss, accuracy_list[epoch], int(lonxitudeDataset * 0.2)))
