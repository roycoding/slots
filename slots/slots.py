"""
slots

A Python library to perform simple multi-armed bandit analyses.

Scenarios:
    - Run MAB test on simulated data (N bandits), default epsilon-greedy test.
        mab = slots.MAB(probs = [0.1,0.15,0.05])
        mab.run(trials = 10000)
        mab.best  # Bandit with highest probability after T trials

    - Run MAB test on "real" payout data (probabilites unknown).
        mab = slots.MAB(hist_payouts = [[0,0,...], [1,0,...], [0,1,...])
        mab.run(trials = 10000)

    - Run MAB test on "live" data
        mab = slots.MAB(num_bandits=3, live=True)
        mab.online_trial(bandit=1, payout=0)
"""

from typing import Optional, List, Dict, Any, Union, Callable

import numpy as np


class MAB(object):
    """
    Multi-armed bandit test class.
    """

    def __init__(
        self,
        num_bandits: Optional[int] = 3,
        probs: Optional[np.ndarray] = None,
        hist_payouts: Optional[List[np.ndarray]] = None,
        live: Optional[bool] = False,
        stop_criterion: Optional[Dict] = {"criterion": "regret", "value": 0.1},
    ) -> None:
        """
        Parameters
        ----------
        num_bandits : int, optional
            default is 3
        probs : array of floats, optional
            payout probabilities
        hist_payouts : list of lists of ints, one array per bandit, optional
            This is for testing on historical data.
            If you set `probs` or `live` is True, `hist_payouts` should be None.
        live : bool, optional
            Whether the use is for a live, online trial.
        stop_criterion : dict, optional
            Stopping criterion (str) and threshold value (float).
        """

        self.choices: List[int] = []

        if not probs:
            if not hist_payouts:
                if live:
                    # Live trial scenario, where nothing is known except the
                    # number of bandits
                    self.bandits: Bandits = Bandits(
                        live=True, payouts=np.zeros(num_bandits)
                    )
                else:
                    # A pure experiment scenario with random probabilities.
                    self.bandits = Bandits(
                        probs=np.random.rand(num_bandits),
                        payouts=np.zeros(num_bandits),
                        live=False,
                    )
            else:
                # Run strategies on known historical sequence of payouts. Probabilities are not known.
                num_bandits = len(hist_payouts)
                if live:
                    print(
                        "slots: Cannot have a defined array of payouts and live=True. live set to False"
                    )
                self.bandits = Bandits(
                    hist_payouts=hist_payouts,
                    payouts=np.zeros(num_bandits),
                    live=False,
                )
        else:
            if hist_payouts:
                # A pure experiment scenario with known historical payout values. Probabilities will be ignored.
                num_bandits = len(probs)
                print(
                    "slots: Since historical payout data has been supplied, probabilities will be ignored."
                )
                if len(probs) == len(hist_payouts):
                    self.bandits = Bandits(
                        hist_payouts=hist_payouts,
                        live=False,
                        payouts=np.zeros(num_bandits),
                    )
                else:
                    raise Exception(
                        "slots: Dimensions of probs and payouts mismatched."
                    )
            else:
                # A pure experiment scenario with known probabilities
                num_bandits = len(probs)
                self.bandits = Bandits(
                    probs=probs, payouts=np.zeros(num_bandits), live=False
                )

        self.wins: np.ndarray = np.zeros(num_bandits)
        self.pulls: np.ndarray = np.zeros(num_bandits)

        # Set the stopping criteria
        self.criteria: Dict[str, Callable[[Optional[float]], bool]] = {
            "regret": self.regret_met
        }
        if not stop_criterion:
            self.criterion: str = "regret"
            self.stop_value: float = 0.1
        else:
            self.criterion = stop_criterion.get("criterion", "regret")
            self.stop_value = stop_criterion.get("value", 0.1)

        # Bandit selection strategies
        self.strategies: List[str] = [
            "eps_greedy",
            "softmax",
            "ucb",
            "bayesian",
        ]

    def run(
        self,
        trials: int = 100,
        strategy: str = "eps_greedy",
        parameters: Optional[Dict] = None,
    ) -> None:
        """
        Run MAB test with T trials.

        Parameters
        ----------
        trials : int
            Number of trials to run.
        strategy : str
            Name of selected strategy. "eps_greedy" is default.
        parameters : dict
            Parameters for selected strategy.

        Available strategies:
            - Epsilon-greedy ("eps_greedy")
            - Softmax ("softmax")
            - Upper confidence bound ("ucb")

        Returns
        -------
        None
        """

        if trials < 1:
            raise Exception(
                "slots.MAB.run: Number of trials cannot be less than 1!"
            )

        else:
            if strategy not in self.strategies:
                raise Exception(
                    "slots.MAB,run: Strategy name invalid. Choose from:"
                    " {}".format(", ".join(self.strategies))
                )

        # Run strategy
        for n in range(trials):
            self._run(strategy, parameters)

    def _run(self, strategy: str, parameters: Optional[Dict] = None) -> None:
        """
        Run single trial of MAB strategy.

        Parameters
        ----------
        strategy : str
        parameters : dict

        Returns
        -------
        None
        """

        choice: int = self.run_strategy(strategy, parameters)
        self.choices.append(choice)
        payout: Optional[int] = self.bandits.pull(choice)
        if payout is None:
            print("Trials exhausted. No more values for bandit", choice)
            return None
        else:
            self.wins[choice] += payout
        self.pulls[choice] += 1

    def run_strategy(
        self, strategy: str, parameters: Optional[Dict] = None
    ) -> int:
        """
        Run the selected strategy and retrun bandit choice.

        Parameters
        ----------
        strategy : str
            Name of MAB strategy.
        parameters : dict
            Strategy function parameters

        Returns
        -------
        int
            Bandit arm choice index
        """

        return self.__getattribute__(strategy)(params=parameters)

    # ###### ----------- MAB strategies ---------------------------------------####
    def max_mean(self) -> int:
        """
        Pick the bandit with the current best observed proportion of winning.

        Returns
        -------
        int
            Index of chosen bandit
        """

        return np.argmax(self.wins / (self.pulls + 0.1))

    def bayesian(self, params: Any = None) -> int:
        """
        Run the Bayesian Bandit algorithm which utilizes a beta distribution
        for exploration and exploitation.

        Parameters
        ----------
        params : None
            For API consistency, this function can take a parameters argument,
            but it is ignored.

        Returns
        -------
        int
            Index of chosen bandit
        """
        p_success_arms: List[float] = [
            np.random.beta(self.wins[i] + 1, self.pulls[i] - self.wins[i] + 1)
            for i in range(len(self.wins))
        ]

        return np.array(p_success_arms).argmax()

    def eps_greedy(self, params: Optional[Dict[str, float]] = None) -> int:
        """
        Run the epsilon-greedy strategy and update self.max_mean()

        Parameters
        ----------
        Params : dict
            Epsilon

        Returns
        -------
        int
            Index of chosen bandit
        """

        default_eps: float = 0.1

        if params and type(params) == dict:
            eps: float = params.get("epsilon", default_eps)
            try:
                float(eps)
            except ValueError:
                print("slots: eps_greedy: Setting eps to default")
                eps = default_eps
        else:
            eps = default_eps

        r: int = np.random.rand()

        if r < eps:
            return np.random.choice(
                list(set(range(len(self.wins))) - {self.max_mean()})
            )
        else:
            return self.max_mean()

    def softmax(self, params: Optional[Dict] = None) -> int:
        """
        Run the softmax selection strategy.

        Parameters
        ----------
        Params : dict
            Tau

        Returns
        -------
        int
            Index of chosen bandit
        """

        default_tau: float = 0.1

        if params and type(params) == dict:
            tau: float = params.get("tau", default_tau)
            try:
                float(tau)
            except ValueError:
                print("slots: softmax: Setting tau to default")
                tau = default_tau
        else:
            tau = default_tau

        # Handle cold start. Not all bandits tested yet.
        if True in (self.pulls < 3):
            return np.random.choice(range(len(self.pulls)))
        else:
            payouts: np.ndarray = self.wins / (self.pulls + 0.1)
            norm: float = sum(np.exp(payouts / tau))

        ps: np.ndarray = np.exp(payouts / tau) / norm

        # Randomly choose index based on CMF
        cmf: List[int] = [sum(ps[: i + 1]) for i in range(len(ps))]

        rand: float = np.random.rand()

        found: bool = False
        found_i: int = 0
        i: int = 0
        while not found:
            if rand < cmf[i]:
                found_i = i
                found = True
            else:
                i += 1

        return found_i

    def ucb(self, params: Optional[Dict] = None) -> int:
        """
        Run the upper confidence bound MAB selection strategy.

        This is the UCB1 algorithm described in
        https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

        Parameters
        ----------
        params : None
            For API consistency, this function can take a parameters argument,
            but it is ignored.

        Returns
        -------
        int
            Index of chosen bandit
        """

        # UCB = j_max(payout_j + sqrt(2ln(n_tot)/n_j))

        # Handle cold start. Not all bandits tested yet.
        if True in (self.pulls < 3):
            return np.random.choice(range(len(self.pulls)))
        else:
            n_tot: int = sum(self.pulls)
            payouts: np.ndarray = self.wins / (self.pulls + 0.1)
            ubcs: np.ndarray = payouts + np.sqrt(
                2 * np.log(n_tot) / self.pulls
            )

            return np.argmax(ubcs)

    # ###------------------------------------------------------------------####

    def best(self) -> Optional[int]:
        """
        Return current 'best' choice of bandit.

        Returns
        -------
        int
            Index of bandit
        """

        if len(self.choices) < 1:
            print("slots: No trials run so far.")
            return None
        else:
            return np.argmax(self.wins / (self.pulls + 0.1))

    def est_probs(self) -> Optional[np.ndarray]:
        """
        Calculate current estimate of average payout for each bandit.

        Returns
        -------
        array of floats or None
        """

        if len(self.choices) < 1:
            print("slots: No trials run so far.")
            return None
        else:
            return self.wins / (self.pulls + 0.1)

    def regret(self) -> float:
        """
        Calculate expected regret, where expected regret is
        maximum optimal reward - sum of collected rewards, i.e.

        expected regret = T*max_k(mean_k) - sum_(t=1-->T) (reward_t)

        Returns
        -------
        float
        """

        return (
            sum(self.pulls) * np.max(np.nan_to_num(self.wins / self.pulls))
            - sum(self.wins)
        ) / sum(self.pulls)

    def crit_met(self) -> bool:
        """
        Determine if stopping criterion has been met.

        Returns
        -------
        bool
        """

        if True in (self.pulls < 3):
            return False
        else:
            return self.criteria[self.criterion](self.stop_value)

    def regret_met(self, threshold: Optional[float] = None) -> bool:
        """
        Determine if regret criterion has been met.

        Parameters
        ----------
        threshold : float

        Returns
        -------
        bool
        """

        if not threshold:
            return self.regret() <= self.stop_value
        elif self.regret() <= threshold:
            return True
        else:
            return False

    # ## ------------ Online bandit testing ------------------------------ ####
    def online_trial(
        self,
        bandit: Optional[int] = None,
        payout: Optional[int] = None,
        strategy: str = "eps_greedy",
        parameters: Optional[Dict] = None,
    ) -> Dict:
        """
        Update the bandits with the results of the previous live, online trial.
            Next run a the selection algorithm. If the stopping criteria is
            met, return the best arm estimate. Otherwise return the next arm to
            try.

        Parameters
        ----------
        bandit : int
            Bandit index of most recent trial
        payout : int
            Payout value of most recent trial
        strategy : string
            Name of update strategy
        parameters : dict
            Parameters for update strategy function

        Returns
        -------
        dict
            Format: {'new_trial': boolean, 'choice': int, 'best': int}
        """

        if bandit is not None and payout is not None:
            self.update(bandit=bandit, payout=payout)
        else:
            raise Exception(
                "slots.online_trial: bandit and/or payout value" " missing."
            )

        if self.crit_met():
            return {
                "new_trial": False,
                "choice": self.best(),
                "best": self.best(),
            }
        else:
            return {
                "new_trial": True,
                "choice": self.run_strategy(strategy, parameters),
                "best": self.best(),
            }

    def update(self, bandit: int, payout: int) -> None:
        """
        Update bandit trials and payouts for given bandit.

        Parameters
        ----------
        bandit : int
            Bandit index
        payout : int (0 or 1)

        Returns
        -------
        None
        """

        self.choices.append(bandit)
        self.pulls[bandit] += 1
        self.wins[bandit] += payout
        self.bandits.payouts[bandit] += payout


class Bandits:
    """
    Bandit class.
    """

    def __init__(
        self,
        payouts: np.ndarray,
        probs: Optional[np.ndarray] = None,
        hist_payouts: Optional[List[np.ndarray]] = None,
        live: bool = False,
    ):
        """
        Instantiate Bandit class, determining
            - Probabilities of bandit payouts
            - Bandit payouts

        Parameters
        ----------
        payouts : array of ints
            Cumulative bandit payouts. `payouts` should start as an N
            length array of zeros, where N is the number of bandits.
        probs: array of floats, optional
            Probabilities of bandit payouts.
        hist_payouts: list of arrays of ints, optional
        live : bool, optional
        """

        if not live:
            self.probs: Optional[np.ndarray] = probs
            self.payouts: np.ndarray = payouts
            self.hist_payouts: Optional[List[np.ndarray]] = hist_payouts
            self.live: bool = False
        else:
            self.live = True
            self.probs = None
            self.payouts = payouts

    def pull(self, i: int) -> Optional[int]:
        """
        Return the payout from a single pull of the bandit i's arm.

        Parameters
        ----------
        i : int
            Index of bandit.

        Returns
        -------
        int or None
        """

        if self.live:
            if len(self.payouts[i]) > 0:
                return self.payouts[i].pop()
            else:
                return None
        elif self.hist_payouts:
            if not self.hist_payouts[i]:
                return None
            else:
                _p: int = self.hist_payouts[i][0]
                self.hist_payouts[i] = self.hist_payouts[i][1:]
                return _p
        else:
            if self.probs is None:
                return None
            elif np.random.rand() < self.probs[i]:
                return 1
            else:
                return 0

    def info(self) -> None:
        pass
