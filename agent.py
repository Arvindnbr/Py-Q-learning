import torch
import random
import numpy as np
from collections import deque
from QNet import QNet, QTrainer, plot
from snake import SnakeGameAgent, Direction, Point, BLOCK_SIZE

MAX_MEMMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class AiAgent:
    
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.8 #discount-rate
        self.memmory = deque(maxlen=MAX_MEMMORY)

        self.model = QNet(11, 512, 3)
        self.trainer = QTrainer(model=self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        lpoint = Point(head.x - BLOCK_SIZE, head.y)
        rpoint = Point(head.x + BLOCK_SIZE, head.y)
        upoint = Point(head.x, head.y - BLOCK_SIZE)
        dpoint = Point(head.x, head.y + BLOCK_SIZE)

        ldir = game.direction == Direction.LEFT
        rdir = game.direction == Direction.RIGHT
        udir = game.direction == Direction.UP
        ddir = game.direction == Direction.DOWN

        state = [
            #straight Danger
            (rdir and game.is_collision(rpoint)) or
            (ldir and game.is_collision(lpoint)) or
            (udir and game.is_collision(upoint)) or
            (ddir and game.is_collision(dpoint)),
            #right danger
            (rdir and game.is_collision(dpoint)) or
            (ldir and game.is_collision(upoint)) or 
            (udir and game.is_collision(rpoint)) or 
            (ddir and game.is_collision(lpoint)),
            #left Danger
            (rdir and game.is_collision(upoint)) or
            (ldir and game.is_collision(dpoint)) or
            (udir and game.is_collision(lpoint)) or 
            (ddir and game.is_collision(rpoint)),

            #move dir
            ldir,
            rdir,
            udir,
            ddir,

            #food
            game.food.x < game.head.x, #food on left
            game.food.x > game.head.x, #food on right
            game.food.y < game.head.y, #food on top
            game.food.y > game.head.y  #food on bottom

        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        memmory = (state, action, reward, next_state, game_over)
        self.memmory.append(memmory)

    def train_long_memmory(self):
        if len(self.memmory) > BATCH_SIZE:
            sample = random.sample(self.memmory, BATCH_SIZE)
        else:
            sample = self.memmory
        
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        

    def train_short_memmory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # initially explore the random moves and lateer exploit the models move once the model stabilizes
        self.epsilon = 70 - self.n_game
        final_move = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move[move] = 1
        return final_move

def train():
    plot_score = []
    plot_ms = []
    total_score = 0
    highest_score = 0

    agent = AiAgent()
    snake = SnakeGameAgent()
    snake.reset()

    while True:
        #old state
        old_state = agent.get_state(snake)

        #action for that
        final_move = agent.get_action(old_state)

        #do the move and get the new state
        reward, game_over, score = snake.play_step(final_move)
        new_state = agent.get_state(snake)

        #train short  memmory
        agent.train_short_memmory(old_state, final_move, reward, new_state, game_over)

        #remember the values
        agent.remember(old_state, final_move, reward, new_state, game_over)

        if game_over:

            snake.reset()
            agent.n_game += 1
            agent.train_long_memmory()

            if score > highest_score:
                highest_score = score
                agent.model.save()

            print('Game', agent.n_game, 'Score', score)

            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_ms.append(mean_score)
            plot(plot_score, plot_ms)

if __name__ == "__main__":
    train()