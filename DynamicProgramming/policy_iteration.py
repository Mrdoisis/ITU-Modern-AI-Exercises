import DP_util
import numpy as np

""" Evaluate a random policy."""
random_policy = DP_util.create_random_policy()
DP_util.evaluate_policy(random_policy)

""" Uncomment to visualize a run."""
# DP_util.agent(DP_util.create_random_policy(), verbose=True)


def policy_iteration(theta=0.01, discount_rate=0.5):
    """"""
    # transition probabilities: p(s', r | s, a)
    # Implemented as a dictionary, with key formatted as:
    #   [next_state, reward, state, acton]
    # Note that the reward is always -1
    #
    # EG. state_transition_probabilities[2, -1, 1, 'E'] == 1, as
    # standing in state 1, and going east (action 'E') will move you to
    # state 2.
    state_transition_probabilities = DP_util.create_probability_map()

    # State transitions - i.e.
    # EG. s_to_sprime[1]['E'] what is the next state, if the agent
    # is in state 1, and moves east (performs action 'E').
    s_to_sprime = DP_util.create_state_to_state_prime_verbose_map()

    """ #1: Initialization 

        V_s is a dictionary that contains the value of each state.
        We will use it to create a better policy which will choose 
        the next state based on the maximum value. Remember the terminal 
        states must always have a value of 0.

        policy is a dictionary of the action probabilities for each state.
    """
    V_s = {i: 0 for i in range(16)}  # Everything zero
    policy = DP_util.create_random_policy()  # Random actions

    done = False
    while not done:
        """ # 2: Policy Evaluation.

            Updates the value function V_s, until the change is smaller than theta.
        """
        accurate = False
        while not accurate:
            delta = 0
            for s in range(16):
                v = V_s[s]
                sums = []
                for action in policy[s]:
                    s_prime = s_to_sprime[s][action]
                    sums.append(state_transition_probabilities[s_prime, -1, s, action] * policy[s][action] * (-1 + discount_rate*V_s[s_prime]))
                new_vs = sum(sums)
                V_s[s] = new_vs
                delta = max(delta, np.abs(v - new_vs))
            if delta < theta:
                accurate = True



        """ YOUR CODE HERE! """


        """ #3: Policy improvement

            Updates the policy if necessary. If the policy is stable (doesn't change)
            set done to True.          
        """
        policy_stable = True

        """ YOUR CODE HERE! """
        for s in range(16):
            old_action = policy[s]
            foo = {}
            for action in policy[s]:
                s_prime = s_to_sprime[s][action]
                foo[action] = state_transition_probabilities[s_prime, -1, s, action] * (-1 + discount_rate * V_s[s_prime])

            max_actions = [k for k,v in foo.iteritems() if v == max(foo.values())]

            new_pol = {a : 1./len(max_actions) if a in max_actions else 0.0 for a in ['N', 'S', 'E', 'W']}

            policy[s] = new_pol
            if not old_action == new_pol:
                policy_stable = False

        if policy_stable:
            done = True

    return V_s, policy


V_s, policy = policy_iteration()

DP_util.evaluate_policy(policy)
DP_util.agent(policy, verbose=False)

