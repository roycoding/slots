'''
slots

A Python library to perform simple multi-armed bandit analyses.

Scenarios:
    - Run MAB test on simulated data (N bandits), default epsilon-greedy test.
        mab = slots.MAB(probs = [0.1,0.15,0.05])
        mab.run(trials = 10000)
        mab.best  # Bandit with highest probability after T trials

    - Run MAB test on "real" payout data (probabilites unknown).
        mab = slots.MAB(payouts = [0,0,0,1,0,0,0,0,0,....])
        mab.run(trials = 10000) # Max is length of payouts
'''


import numpy as np


class MAB(object):
    '''
    Multi-armed bandit test class.
    '''

    def __init__(self, num_bandits=None, probs=None, payouts=None, live=False,
                 stop_criterion={'criterion': 'regret', 'value': 1.0}):
        '''
        Instantiate MAB class, determining
            - Number of bandits
            - Probabilities of bandit payouts
            - Bandit payouts

        Parameters (optional):
            - Number of bandits (used alone) - integer
            - Probabilities of bandit payouts - array of floats
            - Amount of bandit payouts
                - array of floats
                - If 'live' = True, a N*T array of floats indication payout
                    amount per pull for N bandits and T trials
            - Boolean indicating if data is live
            - Dict listing name of stopping criterion and threshold value.
        '''

        default_num_bandits = 3

        self.choices = []

        if not probs:
            if payouts is None:
                if not num_bandits:
                    num_bandits = default_num_bandits
                self.bandits = Bandits(probs=[np.random.rand() for x in
                                       range(num_bandits)],
                                       payouts=np.ones(num_bandits))
            else:
                if live:
                    self.bandits = Bandits(live=True, payouts=payouts,
                                           probs=None)
                else:
                    # Not sure why anyone would do this
                    self.bandits = Bandits(probs=[np.random.rand() for x in
                                           range(len(payouts))],
                                           payouts=payouts)
                num_bandits = len(payouts)
        else:
            if payouts:
                self.bandits = Bandits(probs=probs, payouts=payouts)
                num_bandits = len(payouts)
            else:
                self.bandits = Bandits(probs=probs,
                                       payouts=np.ones(len(probs)))
                num_bandits = len(probs)

        self.wins = np.zeros(num_bandits)
        self.pulls = np.zeros(num_bandits)

        # Set the stopping criteria
        self.criteria = {'regret': self.regret_met}
        if stop_criterion.get('criterion') in self.criteria:
            self.criterion = stop_criterion['criterion']
            if stop_criterion.get('value'):
                self.stop_value = stop_criterion['value']
        else:
            self.criterion = 'regret'
            self.stop_value = 0.1

        # Bandit selection strategies
        self.strategies = ['eps_greedy', 'softmax', 'ucb']

    def run(self, trials=100, strategy=None, parameters=None):
        '''
        Run MAB test with T trials.

        Parameters:
            trials (integer) - number of trials to run.
            strategy (string) - name of selected strategy.
            parameters (dict) - parameters for selected strategy.

        Available strategies:
            - Epsilon-greedy ("eps_greedy")
            - Softmax ("softmax")
            - Upper credibility bound ("ucb")
        '''

        if trials < 1:
            raise Exception('MAB.run: Number of trials cannot be less than 1!')
        if not strategy:
            strategy = 'eps_greedy'
        else:
            if strategy not in self.strategies:
                raise Exception('MAB,run: Strategy name invalid. Choose from:'
                                ' {}'.format(', '.join(self.strategies)))

        # Run strategy
        for n in range(trials):
            self._run(strategy, parameters)

    def _run(self, strategy, parameters=None):
        '''
        Run single trial of MAB strategy.

        Input:
            stategy - function
            parameters - dictionary

        Output:
            None
        '''

        choice = self.run_strategy(strategy, parameters)
        self.choices.append(choice)
        payout = self.bandits.pull(choice)
        if payout is None:
            print('Trials exhausted. No more values for bandit', choice)
            return None
        else:
            self.wins[choice] += payout
        self.pulls[choice] += 1

    def run_strategy(self, strategy, parameters):
        '''
        Run the selected strategy and retrun bandit choice.

        Input:
            strategy - string of strategy name
            parameters - dict of strategy function parameters

        Output:
            integer. Call strategy function, which returns bandit arm choice.
        '''

        return self.__getattribute__(strategy)(params=parameters)

# ###### ----------- MAB strategies ---------------------------------------####
    def max_mean(self):
        """
        Pick the bandit with the current best observed proportion of winning.

        Input: self
        Output: int (index of chosen bandit)
        """

        return np.argmax(self.wins / (self.pulls + 0.1))

    def eps_greedy(self, params):
        '''
        Run the epsilon-greedy MAB algorithm.

        Input: dict of parameters (epsilon)
        Output: None
        '''

        if params and type(params) == dict:
            eps = params.get('epsilon')
        else:
            eps = 0.1

        r = np.random.rand()

        if r < eps:
            return np.random.choice(list(set(range(len(self.wins))) -
                                    {self.max_mean()}))
        else:
            return self.max_mean()

    def softmax(self, params):
        '''
        Run the softmax selection algorithm.

        Input: dict of parameters (tau)
        Output: int (index of chosen bandit)
        '''

        default_tau = 0.1

        if params and type(params) == dict:
            tau = params.get('tau')
            try:
                float(tau)
            except ValueError:
                'slots: softmax: Setting tau to default'
                tau = default_tau
        else:
            tau = default_tau

        # Handle cold start. Not all bandits tested yet.
        if True in (self.pulls < 3):
            return np.random.choice(range(len(self.pulls)))
        else:
            payouts = self.wins / (self.pulls + 0.1)
            norm = sum(np.exp(payouts/tau))

        ps = np.exp(payouts/tau)/norm

        # Randomly choose index based on CMF
        cmf = [sum(ps[:i+1]) for i in range(len(ps))]

        rand = np.random.rand()

        found = False
        found_i = None
        i = 0
        while not found:
            if rand < cmf[i]:
                found_i = i
                found = True
            else:
                i += 1

        return found_i

    def ucb(self, params=None):
        '''
        Run the upper credible bound MAB selection algorithm.

        Input: None (doesn't need parameters)
        Output: int (index of chosen bandit)
        '''

        # UBC = j_max(payout_j + sqrt(2ln(n_tot)/n_j))

        # Handle cold start. Not all bandits tested yet.
        if True in (self.pulls < 3):
            return np.random.choice(range(len(self.pulls)))
        else:
            n_tot = sum(self.pulls)
            payouts = self.wins / (self.pulls + 0.1)
            ubcs = payouts + np.sqrt(2*np.log(n_tot)/self.pulls)

            return np.argmax(ubcs)

    # ###------------------------------------------------------------------####

    def best(self):
        '''
        Return current 'best' choice of bandit.

        Input: self
        Output: integer
        '''

        if len(self.choices) < 1:
            print('slots: No trials run so far.')
            return None
        else:
            return np.argmax(self.wins/(self.pulls+0.1))

    def est_payouts(self):
        '''
        Calculate current estimate of average payout for each bandit.

        Input: Self
        Output: list of floats
        '''

        if len(self.choices) < 1:
            print('slots: No trials run so far.')
            return None
        else:
            return self.wins/(self.pulls+0.1)

    def regret(self):
        '''
        Calculate expected regret, where expected regret is

        expected regret = T*max_k(mean_k) - sum_(t=1-->T) (reward_t)

        Input: None
        Output: float
        '''

        return (sum(self.pulls)*np.max(np.nan_to_num(self.wins/self.pulls)) -
                sum(self.wins)) / sum(self.pulls)

    def crit_met(self):
        '''
        Determine if stopping criterion has been met.

        Output: Boolean
        '''

        return self.criteria[self.criterion](self.stop_value)

    def regret_met(self, threshold=None):
        '''
        Determine if regret criterion has been met.

        Input: float (threshold)
        Output: Boolean
        '''

        if not threshold:
            return self.regret() <= self.stop_value
        elif self.regret() <= threshold:
            return True
        else:
            return False

    # ## ------------ Online bandit testing ------------------------------ ####
    def online_trial(self, bandit=None, payout=None, strategy='eps_greedy',
                     parameters=None):
        '''
        Update the bandits with the results of the previous live, online trial.
            Next run a the selection algorithm. If the stopping criteria is
            met, return the best arm estimate. Otherwise return the next arm to
            try.

        Input:
            bandit - int of bandit index
            payout - float of payout value
            strategy - string name of update strategy
            parameters - dict of parameters for update strategy function

        Output:
            dict - format: {'new_trial': boolean, 'choice': int, 'best': int}
        '''

        if bandit and payout:
            self.update(bandit=bandit, payout=payout)
        else:
            raise Exception('slots.online_trial: bandit and/or payout value'
                            ' missing.')

        if self.crit_met():
            return {'new_trial': False, 'choice': self.best(),
                    'best': self.best()}
        else:
            return {'new_trial': True,
                    'choice': self.run_strategy(strategy, parameters),
                    'best': self.best()}

    def update(self, bandit=None, payout=None):
        '''
        Update bandit trials and payouts for given bandit.

        Input: int (bandit number), float (bandit's payout)
        '''

        self.pulls[bandit] += 1
        self.wins[bandit] += payout
        self.bandits.payouts[bandit] += payout


class Bandits():
    '''
    Bandit class.
    '''

    def __init__(self, probs, payouts, live=False):
        '''
        Instantiate Bandit class, determining
            - Probabilities of bandit payouts
            - Bandit payouts

        Parameters:
            - Probabilities of bandit payouts - array of floats
            - Amount of bandit payouts
                - array of floats
                - If 'live' = True, a N*T array of floats indication payout
                    amount per pull for N bandits and T trials
            - Boolean indicating if data is live
        '''

        if not live:
            # Only use arrays of equal length
            if len(probs) != len(payouts):
                raise Exception('Bandits.__init__: Probability and payouts '
                                'arrays of different lengths!')
            self.probs = probs
            self.payouts = payouts
            self.live = False
        else:
            self.live = True
            self.probs = None
            self.payouts = payouts

    def pull(self, i):
        '''
        Return the payout from a single pull of the bandit i's arm.
        '''

        if self.live:
            if len(self.payouts[i]) > 0:
                return self.payouts[i].pop()
            else:
                return None
        else:
            if np.random.rand() < self.probs[i]:
                return self.payouts[i]
            else:
                return 0.0

    def info(self):
        pass
