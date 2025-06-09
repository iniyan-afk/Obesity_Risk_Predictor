import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data = train_data.replace(['yes', 'no'], [1, 0])
test_data = test_data.replace(['yes', 'no'], [1,0])

train_data[pd.get_dummies(train_data['CAEC']).columns] = pd.get_dummies(train_data['CAEC'])
train_data[pd.get_dummies(train_data['MTRANS']).columns] = pd.get_dummies(train_data['MTRANS'])
train_data[pd.get_dummies(train_data['CALC']).columns] = pd.get_dummies(train_data['CALC'])
test_data[pd.get_dummies(test_data['CAEC']).columns] = pd.get_dummies(test_data['CAEC'])
test_data[pd.get_dummies(test_data['MTRANS']).columns] = pd.get_dummies(test_data['MTRANS'])
test_data[pd.get_dummies(test_data['CALC']).columns] = pd.get_dummies(test_data['CALC'])
train_data = train_data.replace(['Male', 'Female', True, False], [1, 0, 1, 0])
test_data = test_data.replace(['Male', 'Female', True, False], [1, 0, 1, 0])

X = train_data.drop(['id', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad'], axis = 1).to_numpy()
X_test = test_data.drop(['id', 'CAEC', 'CALC', 'MTRANS'], axis = 1).to_numpy()

y = train_data['NObeyesdad']
y_num = y.replace(['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'], [0, 1, 2, 3, 4, 5, 6]).to_numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.from_numpy(X).to(torch.float32).to(device)
X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
y_num = torch.from_numpy(y_num).to(torch.long).to(device)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(22, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 7)
        self.relu = nn.ReLU()
    def forward(self, X):
        X = self.l1(X)
        X = self.relu(X)
        X = self.l2(X)
        X = self.relu(X)
        X = self.l3(X)
        X = self.relu(X)
        X = self.l4(X)
        return X
model = Classifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
dataset = TensorDataset(X, y_num)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
epochs = 500
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, label in loader:
        y_pred = model(data)
        loss = loss_fn(y_pred, label)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss/len(loader)
    print("Epoch:", epoch, "Loss:", avg_loss)

y_pred = model(X_test)
y_preds = torch.nn.functional.softmax(y_pred)
y_predictions = y_preds.argmax(dim=1)
y_predictor = y_predictions.cpu().detach().numpy()

obese = pd.DataFrame({'NObeyesdad':y_predictor})
obese = obese.replace([0, 1, 2, 3, 4, 5, 6], ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'])

out = pd.DataFrame({'id':test_data['id'], 'NObeyesdad':obese['NObeyesdad']})
out.to_csv('output.csv', index=False)

