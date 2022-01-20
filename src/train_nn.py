
import torch
from torch.utils.data import DataLoader
from data.dataset import reactionDataset
from models.neural_network import simpleNet
from utils.train_utils import evaluate_model
from utils.train_utils import EarlyStopping
from utils.train_utils import print_model_accuracy
from utils.train_utils import save_model
import torch.optim as optim
import torch.nn as nn

torch.set_printoptions(linewidth = 1000)

BATCH_SIZE = 512
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EPOCHS = 100000
ES_PATIENCE = 10
ES_DELTA = 0
REPRESENTATION = 'chembl'

# ----- set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on: ", DEVICE)

# ----- load datasets
train_dataset = reactionDataset('train', representation = REPRESENTATION)
val_dataset = reactionDataset('val', representation = REPRESENTATION)
test_dataset = reactionDataset('test', representation = REPRESENTATION)

train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)

# ----- load model
model = simpleNet(representation = REPRESENTATION)
model = model.double().to(DEVICE)

# train settings
criterion = nn.BCELoss()
early_stopping = EarlyStopping(patience = ES_PATIENCE , delta = ES_DELTA)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):

        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print prediction accuracy
        print_model_accuracy(outputs, labels, loss, threshold = 0.5, mode = "train")

    if (epoch + 1) % 5 == 0:

        val_losses_avg = evaluate_model(model, criterion, val_loader, DEVICE, epoch)
        stop = early_stopping(val_losses_avg, model)

        if stop:
            print('INFO: Early stopped - val_loss_min: {}'.format(early_stopping.val_loss_min.item()))
            model.load_state_dict(early_stopping.saved_state_dict)
            early_stop = True

save_model(model, "models/best.pt")
print('Finished Training')
