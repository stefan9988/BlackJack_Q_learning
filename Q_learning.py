import random
import numpy as np
from tqdm import tqdm

# Function to calculate the total value of a hand
def calculate_hand_value(hand):
    value = 0
    num_aces = 0
    for card in hand:
        if card in ['J', 'Q', 'K']:
            value += 10
        elif card == 'A':
            value += 11
            num_aces += 1
        else:
            value += int(card)

    # Adjust the value if there are aces and the total value exceeds 21
    while value > 21 and num_aces > 0:
        value -= 10
        num_aces -= 1

    return value

# Function to deal a new card from the deck
def deal_card():
    deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    return random.choice(deck)

def calculate_win_rate(rewards,num):
    c=0
    for i in rewards:
        if i==1:
            c+=1
    return c/num*100

def calculate_best_policy():
    p=[]
    best_policy={}
    for i in range(4,22):
        p.append(np.argmax([q_table[i]['hit'],q_table[i]['stand']]))
    for i in range(18):
        if p[i]==0:
            best_policy[i+4]='hit'
        else:
            best_policy[i+4] = 'stand'
    return best_policy

def epsilon_decay(initial_epsilon, final_epsilon, decay_rate, current_step):
    epsilon = final_epsilon + (initial_epsilon - final_epsilon) * np.exp(-decay_rate * current_step)
    return epsilon

# Define the action space
actions = ['hit', 'stand']

# Define the Q-table
q_table = {}
rewards=[]

# Define the hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
initial_epsilon = 1.0   # Exploration rate
final_epsilon = 0.01
decay_rate = 0.0001

# Game loop
game_over = False

def episode(game_over,step):
    # Initialize the game
    player_hand = []
    dealer_hand = []

    epsilon = epsilon_decay(initial_epsilon, final_epsilon, decay_rate, step)

    # Deal two cards to the player and dealer
    player_hand.append(deal_card())
    player_hand.append(deal_card())
    dealer_hand.append(deal_card())
    dealer_hand.append(deal_card())
    while not game_over:
        player_score = calculate_hand_value(player_hand)

        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            if calculate_hand_value(player_hand) in q_table:
                action = max(q_table[calculate_hand_value(player_hand)], key=q_table[calculate_hand_value(player_hand)].get)
            else:
                action = random.choice(actions)

        # Take action
        if action == 'hit':
            player_hand.append(deal_card())
            player_score = calculate_hand_value(player_hand)

            # Check if player busts
            if player_score > 21:
                game_over = True
                reward = -1
            else:
                reward = 0
        else:  # Stand
            # Dealer's turn
            while calculate_hand_value(dealer_hand) < 17:
                dealer_hand.append(deal_card())

            dealer_score = calculate_hand_value(dealer_hand)

            # Compare scores
            if dealer_score > 21 or player_score > dealer_score:
                reward = 1
            elif dealer_score > player_score:
                reward = -1
            else:
                reward = 0

            game_over = True

        # Update Q-table
        #state = (tuple(player_hand), dealer_hand[0])
        state = calculate_hand_value(player_hand)
        if state not in q_table:
            q_table[state] = {'hit': 0, 'stand': 0}

        max_next_q = max(q_table[state].values())

        q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * max_next_q)

        rewards.append(reward)

if __name__ == '__main__':
    num_of_episodes=100000
    for i in tqdm(range(num_of_episodes)):
        episode(game_over,i)
        game_over = False

    print('Best policy:')
    print(calculate_best_policy())
    win_rate=calculate_win_rate(rewards,num_of_episodes)
    print('Win rate: '+str(win_rate)+' %')


