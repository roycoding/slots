#Multi-armed bandit library notes

### What does the library need to do?
1. Set up N bandits with probabilities, p_i, and payouts, pay_i.
2. Implement several MAB strategies, with kwargs as parameters, and consistent API.
3. Allow for T trials.
4. Continue with more trials (i.e. save state after trials).
5. Values to save:
    1. Current choice
    2. number of trials completed for each arm
    3. scores for each arm
    4. average payout per arm (payout*wins/trials?)
6. Use sane defaults.
7. Be obvious and clean.

###Library API ideas:
Creating a MAB test instance:

```Python
# Default: 3 bandits with random p_i and pay_i = 1
mab = slots.MAB()

# Set up 4 bandits with random p_i and pay_i
mab = slots.MAB(4)

# 4 bandits with specified p_i
mab = slots.MAB(probs = [0.2,0.1,0.4,0.1])

# 3 bandits with specified pay_i
mab = slots.MAB(payouts = [1,10,15])

# Bandits with payouts specified by arrays (i.e. payout data with unknown probabilities)
# payouts is an N * T array, with N bandits and T trials
mab = slots.MAB(live = True, payouts = [[0,0,0,0,1.2,0,0],[0,0.1,0,0,0.1,0.1,0]]
```

Running tests with strategy, S

```Python
# Default: Epsilon-greedy, epsilon = 0.1, num_trials = 1000
mab.run()

# Run chosen strategy with specified parameters and trials
map.eps_greedy(eps = 0.2, trials = 10000)
map.run(strategy = 'eps_greedy',params = {'eps':0.2}, trials = 10000)

# Run strategy, updating old trial data
map.run(continue = True)
```

Displaying / retrieving bandit properties

```Python
# Default: display number of bandits, probabilities and payouts
mab.bandits.info()

# Display info for bandit i
mab.bandits[i]

# Retrieve bandits' payouts, probabilities, etc
mab.bandits.payouts
mab.bandits.probs

# Retrieve count of bandits
mab.bandits.count
```

Setting bandit properties

```Python
# Reset bandits to defaults
map.bandits.reset()

# Set probabilities or payouts
map.bandits.probs_set([0.1,0.05,0.2,0.15])
map.bandits.payouts_set([1,1.5,0.5,0.8])
```

Displaying / retrieving test info

```Python
# Retrieve current "best" bandit
mab.best()

# Retrieve bandit probability estimates
map.prob_est()

# Retrieve bandit probability estimate of bandit i
map.prob_est(i)

# Retrieve bandit payout estimates (p * payout)
map.payout_est()

# Retrieve current bandit choice
map.current()

# Retrieve sequence of choices
map.choices

# Retrieve probabilty estimate history
map.prob_est_sequence

# Retrieve test strategy info (current strategy) -- a dict
map.strategy_info()
```
