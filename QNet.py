import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython import display
import matplotlib.pyplot as plt


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name ='model.pth'):
        out = './models'
        if not os.path.exists(out):
            os.makedirs(out)

        file_name = os.path.join(out,file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.crieterion = nn.MSELoss()

    def train_step(self, state, action, reward, nxt_state, game_over):
        
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        nxt_state = torch.tensor(nxt_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            nxt_state = torch.unsqueeze(nxt_state, 0)
            game_over = (game_over, )

        pred = self.model(state)

        #copy the pred
        target = pred.clone()
        for idx in range(len(game_over)):
            QNew = reward[idx]
            if not game_over[idx]:
                QNew = reward[idx] + self.gamma * torch.max(self.model(nxt_state[idx]))
            
            target[idx][torch.argmax(action).item()] = QNew

        self.optimizer.zero_grad()
        loss = self.crieterion(target, pred)
        loss.backward()

        self.optimizer.step() 


plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.ylim(ymin=0)

    
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.legend()  
    plt.pause(0.1)  
