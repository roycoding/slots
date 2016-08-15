# slots
## Multi-armed bandit library in Python

## Documentation
This documents details the current and planned API for slots. Non-implemented features are noted as such.

### What does the library need to do? An aspirational list.
1. Set up N bandits with probabilities, p_i, and payouts, pay_i.
2. Implement several MAB strategies, with kwargs as parameters, and consistent API.
3. Allow for T trials.
4. Continue with more trials (i.e. save state after trials).
5. Values to save:
    1. Current choice
    2. number of trials completed for each arm
    3. scores for each arm
    4. average payout per arm (payout*wins/trials?)
    5. Current regret.  Regret = Trials*mean_max - sum^T_t=1(reward_t)
        - See [ref](http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf)
6. Use sane defaults.
7. Be obvious and clean.

### Library API ideas:
#### Running slots with a live website
```Python
# Using slots to determine the best of 3 variations on a live website. 3 is the default.
mab = slots.MAB(3)

# Make the first choice randomly, record responses, and input reward
# 2 was chosen.
# Run online trial (input most recent result) until test criteria is met.
mab.online_trial(bandit=2,payout=1)

# Repsonse of mab.online_trial() is a dict of the form:
{'new_trial': boolean, 'choice': int, 'best': int}

# Where:
#   If the criterion is met, new_trial = False.
#   choice is the current choice of arm to try.
#   best is the current best estimate of the highest payout arm.
```

#### Creating a MAB test instance:

```Python
# Default: 3 bandits with random p_i and pay_i = 1
mab = slots.MAB(live=False)

# Set up 4 bandits with random p_i and pay_i
mab = slots.MAB(4, live=False)

# 4 bandits with specified p_i
mab = slots.MAB(probs = [0.2,0.1,0.4,0.1], live=False)

# 3 bandits with specified pay_i
mab = slots.MAB(payouts = [1,10,15], live=False)
```

#### Running tests with strategy, S

```Python
# Default: Epsilon-greedy, epsilon = 0.1, num_trials = 100
mab.run()

# Run chosen strategy with specified parameters and number of trials
mab.run(strategy = 'eps_greedy',params = {'eps':0.2}, trials = 10000)

# Run strategy, updating old trial data
# (NOT YET IMPLEMENTED)
mab.run(continue = True)
```

#### Displaying / retrieving bandit properties

```Python
# Default: display number of bandits, probabilities and payouts
# (NOT YET IMPLEMENTED)
mab.bandits.info()

# Display info for bandit i
# (NOT YET IMPLEMENTED)
mab.bandits[i]

# Retrieve bandits' payouts, probabilities, etc
mab.bandits.payouts
mab.bandits.probs

# Retrieve count of bandits
# (NOT YET IMPLEMENTED)
mab.bandits.count
```

#### Setting bandit properties

```Python
# Reset bandits to defaults
# (NOT YET IMPLEMENTED)
mab.bandits.reset()

# Set probabilities or payouts
# (NOT YET IMPLEMENTED)
mab.bandits.probs_set([0.1,0.05,0.2,0.15])
mab.bandits.payouts_set([1,1.5,0.5,0.8])
```

#### Displaying / retrieving test info

```Python
# Retrieve current "best" bandit
mab.best()

# Retrieve bandit probability estimates
# (NOT YET IMPLEMENTED)
mab.prob_est()

# Retrieve bandit probability estimate of bandit i
# (NOT YET IMPLEMENTED)
mab.prob_est(i)

# Retrieve bandit payout estimates (p * payout)
mab.est_payout()

# Retrieve current bandit choice
# (NOT YET IMPLEMENTED, use mab.choices[-1])
mab.current()

# Retrieve sequence of choices
mab.choices

# Retrieve probability estimate history
# (NOT YET IMPLEMENTED)
mab.prob_est_sequence

# Retrieve test strategy info (current strategy) -- a dict
# (NOT YET IMPLEMENTED)
mab.strategy_info()
```

### Proposed MAB strategies
- [x] Epsilon-greedy
- [ ] Epsilon decreasing
- [x] Softmax
- [ ] Softmax decreasing
- [x] Upper credible bound
- [x] Bayesian bandits
