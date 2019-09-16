import numpy as np
import matplotlib.pyplot as plt

from cliff import Cliff
import QL_utils

class TabularNStepQLearning:
    def __init__(self, state_shape, num_actions, n=1):
        self.num_actions = num_actions
        self.n = n  # n parameter in n-step QLearning (ignore for now)

        self.eps = 0.1  # Epsilon parameter - play around with it!
        self.alpha = 0.1  # learning rate
        self.gamma = 0.99  # discount factor

        self.exp = [] # R

        # Qtable for storing Qvalues. We create one indes for each state-action pair
        self.tab_shape = np.hstack([state_shape, num_actions])
        self.Qtable = np.zeros(self.tab_shape)

    def action(self, state):
        """ With probability 1-eps: Retur the expected optimal action, given the current state.
            With probability eps: return a random action.
        """
        """ YOUR CODE HERE"""
        rn = np.random.random()
        if rn < self.eps:
            # Choose random action
            legal_actions = [0, 1, 2, 3]
            return np.random.choice(legal_actions)
        else:
            # Choose best action
            actions = self.Qtable[tuple(state)]
            best_action = np.argmax(actions)
            return best_action

    def compute_G(self):
        """ Returns the discounted reward.
        """
        """ YOUR CODE HERE"""
        G = 0
        k = 0
        for i in self.exp:
            (s, a, r, s_, a_, d) = i
            g = self.Qtable[tuple(s)][a]
            G += pow(self.gamma, k) * g
            k += 1


        return G

    def update(self, s, a, r, s_, a_, d):
        """ Given (state, action, reward, next_state, next_action, done),
            update the self.Qtable.
        """
        """ YOUR CODE HERE"""
        self.exp.append([s, a, r, s_, a_, d])
        q_now = self.Qtable[tuple(s)][a]
        max_n = max(self.Qtable[tuple(s_)])
        temporal_diff = r + self.gamma * max_n - q_now
        self.Qtable[tuple(s)][a] = q_now + self.alpha * temporal_diff

action_dict = {0:"Up", 1:"Right", 2:"Down", 3:"Left"}

def run_loop(env, agent, title, max_e=1000, render=False, update=True, plot_frequency=5e3):
    t = 0; i = 0; e = 0
    s, r, d, _ = env.reset()
    a_ = agent.action(s)
    ep_lens = []; rewards = []
    r_sum = 0
    since_last_plot = 0

    while True:
        i += 1; t += 1; since_last_plot += 1
        a = a_
        s_, r, d, _ = env.step(a)
        a_ = agent.action(s_)

        if update:
            agent.update(s=s, a=a, r=r, s_=s_, a_=a_, d=d)
        r_sum += r
        s = np.copy(s_)

        if render:
            QL_utils.render_helper(env, title, i)

        if d or i > 1e6:
            if since_last_plot > plot_frequency:
                since_last_plot = 0
                QL_utils.plot_helper(title, e, agent, env)

            ep_lens.append(i)
            rewards.append(r_sum)
            r_sum = 0; e += 1; i = 0
            s, r, d, _ = env.reset()
            a_ = agent.action(s)

        if max_e and e >= max_e:
            break

    return ep_lens, rewards


if __name__ == '__main__':
    ## Run settings
    num_runs = 10  # Number of runs to average rewards over
    n = 1  # n parameter in n-step Bootstrapping

    ## Q-learning
    TN_QLearning_rewards = []
    env = Cliff()
    for i in range(num_runs):
        # Create agent
        TN_QLearning = TabularNStepQLearning(env.state_shape, env.num_actions, n=n)

        # Run training loop
        _, rewards = run_loop(env, TN_QLearning,  str(n)+'-step QLearning, run: ' + str(i))
        TN_QLearning_rewards.append(rewards)
    TN_QLearning_rewards = np.array(TN_QLearning_rewards)

    # Run the last QLearning agent using visualizations.
    # Try running this a couple of times
    run_loop(env, TN_QLearning, 'QLearning, n='+str(n), max_e=1, render=True)


    # Plot the rewards
    plt.figure()
    include_sd = False # include standard deviation in plot
    QL_utils.reward_plotter(TN_QLearning_rewards, 'QLearning', 'r', include_sd=include_sd, smooth_factor=2)

    axes = plt.gca()
    axes.set_ylim([-100, 0])
    plt.show()
