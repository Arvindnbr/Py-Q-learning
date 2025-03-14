# Py-Q-learning
Develop and implement Reinforcement Learning + Deep Q Learning using Pytorch

Actions:
Actions available to the agents;
- straight - [1,0,0]
- right - [0,1,0]
- left - [0,0,1]

Rewards: 
we set the following as the rewards fro out agent
- eat red +5
- game-over -5
- else 0

State:
reminds the model weather there is food neaby or there is danger ahead.
has 11 different states {
- direction: direction_left, direction_right, direction_up, direction_down
- danger: danger_straight, danger_left, danger_right
- food: food_left, food_up, food_right, food_down
}


the model takes state as the input and returns the corresponding action and it remembers the previous action and state while going for the next.
[11 bool val] -> [1,0,0]

we initiate with random Q(quality of action) value
iterate over the action = model.pred(state)
do the action
measure reward
update Q value and train


the loss used here is Bellman eqn derived mse
