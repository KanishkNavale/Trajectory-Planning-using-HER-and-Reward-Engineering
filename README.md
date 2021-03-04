# Trajectory Planning using HER & Reward Engineering
Trajectory planning based on Reinforcement Learning with Hindsight Experience Replay and Dense Reward Engineering to solve openai-gym robotics "FetchReach-v1" environment using TensorFlow.

## Reinforcement Algorithms,
1. Dense Reward Engineering <br />
    Augmented the sparse reward into dense rewards. Used the initial EOAT position to map the distance with Target position. As the distance reduces the reward increases. Rescaled the reward into a range of [-1,1].

2. Hindsight Experience Relay (HER) <br />
    Tricked the environment that it reached the goal. 

3. HER + Dense Reward Engineering <br />
    Combination of (1) and (2)

## Training History,
* Measure: Accumulated Rewards over Episodes
1. Dense Reward Engineering,
<p ><img src="Test Analysis/Reward Engg: Cummulated Rewards vs Episodes.png" width="500" ></p>

2. HER,
<p ><img src="Test Analysis/HER: Cummulated Rewards vs Episodes.png" width="500" ></p>

3. HER + Dense Reward Engineering,
<p ><img src="Test Analysis/HER + Reward Engg: Cummulated Rewards vs Episodes.png" width="500" ></p>

## Dependencies
Install dependencies using:
```bash
pip3 install -r requirements.txt 
```
1. Additionally install 'mujoco_py' according to 'https://github.com/openai/mujoco-py'

## Challenge
* Achieve the goal position with minimum episode steps

## Contact
* email: navalekanishk@gmail.com