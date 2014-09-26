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

class MAB():
    '''
    Multi-armed bandit test class.
    '''

    def __init__(self, num_bandits=None,probs=None,payouts=None,live=False):
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
        '''

        default_num_bandits = 3

        self.choices = []

        if not probs:
            if payouts is None:
                if not num_bandits:
                    num_bandits = default_num_bandits
                self.bandits = Bandits(probs = [np.random.rand() for x in 
                                            range(num_bandits)],
                                            payouts = np.ones(num_bandits))
            else:
                if live:
                    self.bandits = Bandits(live = True, payouts = payouts, probs=None)
                else:
                    # Not sure why anyone would do this
                    self.bandits = Bandits(probs = [np.random.rand() for x in 
                                            range(len(payouts))],
                                            payouts = payouts)
                num_bandits = len(payouts)
        else:
            if payouts:
                self.bandits = Bandits(probs = probs, payouts = payouts)
                num_bandits = len(payouts)
            else:
                self.bandits = Bandits(probs = probs,
                                        payouts = np.ones(len(payouts)))
                num_bandits = len(probs)


        self.wins = np.zeros(num_bandits)
        self.pulls = np.zeros(num_bandits)

    def run(self, trials=100, strategy=None, parameters=None):
        '''
        Run MAB test with T trials.
        
        Paramters:
            trials (integer) - number of trials to run.
            strategy (string) - name of selected strategy.
            parameters (dict) - parameters for selected strategy.

        Currently on epsilon greedy is implemented.
        '''

        strategies = {'eps_greedy':self.eps_greedy}

        if trials < 1:
            raise Exception('MAB.run: Number of trials cannot be less than 1!')
        if not strategy:
            strategy = 'eps_greedy'
        else:
            if strategy not in strategies:
                raise Exception('MAB,run: Strategy name invalid. Choose from: '
                                 + ', '.join(strategies))

        # Run strategy
        for n in xrange(trials):
            choice = strategies[strategy](params=parameters)
#            print 'DEBUG - run - choice:',choice
            self.choices.append(choice)
            payout = self.bandits.pull(choice)
            if payout is None:
                print 'Trials exhausted. No more values for bandit',choice
                break
            else:
                self.wins[choice] += payout
            self.pulls[choice] += 1
#            print 'DEBUG - run - choices:',self.choices
#            print 'DEBUG - run - pulls:',self.pulls
#            print 'DEBUG - run - wins:',self.wins

    def max_mean(self):
        """
        Pick the bandit with the current best observed proportion of winning 

        Input: self
        Output: None
        """
        return np.argmax( self.wins / ( self.pulls +1 ) )

    def eps_greedy(self,params):
        '''
        Run the epsilon-greedy MAB algorithm.

        Input: dict of parameters (epsilon)
        Output: None
        '''

        if params and type(params) == dict:
            eps = param
        else:
            eps = 0.1

        r = np.random.rand()
        if r < eps:
            return np.random.choice(list(set(range(len(self.wins)))-{self.max_mean()}))
        else:
            return self.max_mean()

    def best(self):
        '''
        Return current 'best' choice of bandit.

        Input: self
        Output: integer
        '''

        if len(self.choices) < 1:
            print 'slots: No trials run so far.'
            return None
        else:
            return np.argmax(self.wins/self.pulls)

    def est_payouts(self):
        '''
        Calculate current estimate of average payout for each bandit.

        Input: Self
        Output: list of floats
        '''

        if len(self.choices) < 1:
            print 'slots: No trials run so far.'
            return None
        else:
            return self.wins/self.pulls

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
                raise Exception('Bandits.__init__: Probability and payouts arrays of different lengths!')
            self.probs = probs
            self.payouts = payouts
            self.live = False
        else:
            self.live = True
            self.probs = None
            self.payouts = payouts

    def pull(self,i):
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
