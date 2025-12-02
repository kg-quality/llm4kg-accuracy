import random
import numpy as np
import networkx as ntx

from tqdm import tqdm
from scipy import stats, optimize
from .stratificationStrategies import stratifyCSRF


def clusterCostFunction(heads, triples, c1=45, c2=25):
    """
    Compute the cluster-based annotation cost function (in hours)

    :param heads: num of heads (clusters)
    :param triples: num of triples
    :param c1: average cost for Entity Identification (EI)
    :param c2: average cost for Fact Verification (FV)
    :return: the annotation cost function (in hours)
    """

    return (heads * c1 + triples * c2) / 3600


def computeOracleFeature(clusters, groundTruth):
    """
    Compute (oracle) accuracy for clusters -- only used to compute stratified sampling lower bound

    :param clusters: KG clusters (star-shaped clusters)
    :param groundTruth: KG ground truth (triple-based)
    :return: average accuracy for clusters
    """

    # compute cluster accuracy for every cluster in clusters
    accs = [sum([groundTruth[triple] for triple in cluster]) / len(cluster) for cluster in clusters.values()]
    return accs


def computeLLMFeature(preds, targetLLM, clusters):
    """
    Compute LLM accuracy for clusters based on LLM of choice

    :param preds: LLM predictions on fact correctness ({LLM1: {fact1: pred, ...}, LLM2: {fact1: pred, ...}, ...})
    :param targetLLM: target LLM
    :param clusters: KG clusters (star-shaped clusters)
    :return: LLM accuracy for clusters
    """

    # setup label encoding
    label2value = {'false': 0, 'true': 1}

    # iterate over LLM predictions and store them for each fact
    fact2value = {id_: label2value[label] for id_, label in preds[targetLLM].items()}

    cluster2acc = {}
    for head, cluster in clusters.items():  # iterate over cluster triples and compute LLM-derived cluster accuracies
        accLLM = 0
        for fact in cluster:  # sum LLM prediction for each fact in cluster
            accLLM += fact2value[fact]
        # obtain the LLM cluster accuracy
        accLLM /= len(cluster)
        cluster2acc[head] = accLLM
    return list(cluster2acc.values())


def computeLLMAggFeature(preds, clusters):
    """
    Compute LLM (aggregated) score for clusters

    :param preds: LLM predictions on fact correctness ({LLM1: {fact1: pred, ...}, LLM2: {fact1: pred, ...}, ...})
    :param clusters: KG clusters (star-shaped clusters)
    :return: LLM (aggregated) accuracy for clusters
    """

    # setup label encoding
    label2value = {'false': 0, 'true': 1}

    fact2value = {}
    for model in preds.values():  # iterate over LLM predictions and store cumulative score for each fact
        for id_, label in model.items():
            if id_ in fact2value:  # already seen fact -- cumulate the score
                fact2value[id_] += label2value[label]
            else:  # unseen new fact -- store the first score
                fact2value[id_] = label2value[label]

    cluster2agg = {}
    for head, cluster in clusters.items():  # iterate over cluster triples and compute LLM-derived cluster accuracies
        aggLLM = 0
        for fact in cluster:  # sum aggregated score for each fact (aggregated over the number of considered LLMs)
            aggLLM += fact2value[fact]/len(preds)
        # obtain the LLM (aggregated) cluster score
        aggLLM /= len(cluster)
        cluster2agg[head] = aggLLM
    return list(cluster2agg.values())


def computeRandomFeature(clusters, coin_prob=0.5, num_coins=1):
    """
    Compute random accuracy for clusters

    :param clusters: KG clusters (star-shaped clusters)
    :param coin_prob: Coin flip probability
    :param num_coins: Number of coins used
    :return: Random (average) accuracy for clusters
    """

    cluster2rnd = {}
    for head, cluster in tqdm(clusters.items(), desc='Generating random annotations...'):  # iterate over cluster triples and compute random cluster accuracies
        annots = np.random.choice([0, 1], size=(len(cluster), num_coins), p=[coin_prob, 1 - coin_prob])
        accR = np.mean(annots, axis=1)
        cluster2rnd[head] = np.mean(accR)
    return list(cluster2rnd.values())


class SRSSampler(object):
    """
    This class represents the Simple Random Sampling (SRS) scheme used to perform KG accuracy evaluation.
    The SRS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05, a_prior=1, b_prior=1):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability αlpha/2

        :param alpha: the user defined confidence level
        :param a_prior: the user defined alpha prior for bayesian credible intervals
        :param b_prior: the user defined beta prior for bayesian credible intervals
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

        # bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # possible 1-alpha intervals
        self.computeMoE = {'bayesHPD': self.computeBHPD}

    def updatePriors(self, a_prior, b_prior):
        """
        Update alpha and beta priors for bayesian credible intervals

        :param a_prior: the new alpha prior
        :param b_prior: the new beta prior
        """

        # update bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

    def estimate(self, sample):
        """
        Estimate the KG accuracy based on sample

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: KG accuracy estimate
        """

        return sum(sample)/len(sample)

    def computeVar(self, sample):
        """
        Compute the sample variance

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: sample variance
        """

        # estimate mean
        ae = self.estimate(sample)
        # count number of clusters in sample
        n = len(sample)
        # compute variance
        var = (1/n) * (ae * (1-ae))
        return var

    def computeBHPD(self, sample):
        """
        Compute the Bayesian Highest Posterior Density Credible Interval (CrI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute sample size and number of successes
        n = len(sample)
        x = sum(sample)

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if x == 0:  # posterior distr has mode in x = 0 -- HPD interval is (0, q(1-alpha))
            return 0, posterior.ppf(1 - self.alpha)

        if x == n:  # posterior distr has mode in x = n -- HPD interval is (q(alpha), 1)
            return posterior.ppf(self.alpha), 1

        # compute credible mass
        mass = 1 - self.alpha

        # objective function to minimize -- i.e. interval width
        def objective(params):
            lower, upper = params
            return upper - lower

        # constraint: interval should contain credible mass
        def constraint(params):
            lower, upper = params
            return posterior.cdf(upper) - posterior.cdf(lower) - mass

        # initial guess for the interval -- rely on corresponding ET CrI (w/o ad hoc changes)
        guess = posterior.ppf([self.alpha/2, 1-self.alpha/2])

        # minimize the width of the interval (objective) subject to the credible mass (constraint)
        res = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)])

        if res.success:  # optimization succeeded -- return the derived HPD CrI
            return res.x
        else:  # optimization failed -- increase the number of performed iterations to 1000 -- default = 100
            print('optimization failed w/ standard number of iterations (100) -- increase to 1000 iterations and retry')
            rres = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)], options={'maxiter': 1000})
            if rres.success:  # optimization succeeded -- return the derived HPD CrI
                print('optimization succeeded w/ 1000 iterations')
                return rres.x
            else:  # optimization failed -- unexpected behavior, raise exception
                raise RuntimeError('optimization failed w/ 1000 iterations -- unexpected behavior, inspection required')

    def run(self, kg, groundTruth, minSample=30, thrMoE=0.05, ciMethod='bayesHPD', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ SRS and stop when MoE <= thr

        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute 1-alpha Intervals (CIs)
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        if self.a_prior == -1 and self.b_prior == -1 and ciMethod == 'bayesHPD':
            adaptive = True
        else:
            adaptive = False

        estimates = []
        for _ in tqdm(range(iters)):  # perform iterations
            # set params
            lowerB = 0.0
            upperB = 1.0
            heads = {}
            sample = []

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than or equal to thrMoE
                # perform SRS over the KG
                id_, triple = random.choices(population=kg, k=1)[0]
                if triple[0] not in heads:  # found new head (cluster) -- increase the num of clusters within sample
                    heads[triple[0]] = 1
                # get annotations for triples within sample
                sample += [groundTruth[id_]]

                if len(sample) >= minSample:  # compute CI
                    if adaptive:  # conduct adaptive strategy
                        # set temporary interval boundaries
                        lowerMin = 0.0
                        upperMin = 1.0
                        for prior in [1/3, 1/2, 1]:  # iterate over non-informative priors
                            # update prior
                            self.updatePriors(prior, prior)
                            # compute interval given prior
                            lowerCurrent, upperCurrent = self.computeMoE[ciMethod](sample)
                            if upperCurrent-lowerCurrent < upperMin-lowerMin:  # computed interval has smaller width than min -- update
                                lowerMin = lowerCurrent
                                upperMin = upperCurrent
                        # set interval w/ minimal boundaries
                        lowerB = lowerMin
                        upperB = upperMin
                    else:
                        lowerB, upperB = self.computeMoE[ciMethod](sample)

            # compute cost function
            cost = clusterCostFunction(len(heads), len(sample), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[len(sample), estimate, cost, lowerB, upperB]]
        # return stats
        return estimates


class TWCSSampler(object):
    """
    This class represents the Two-stage Weighted Cluster Sampling (TWCS) scheme used to perform KG accuracy evaluation.
    The TWCS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05, a_prior=1, b_prior=1):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        :param a_prior: the user defined alpha prior for bayesian credible intervals
        :param b_prior: the user defined beta prior for bayesian credible intervals
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

        # bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # instantiate the SRS sampling method
        self.srs = SRSSampler(alpha=self.alpha, a_prior=self.a_prior, b_prior=self.b_prior)

        # possible 1-alpha intervals
        self.computeMoE = {'bayesHPD': self.computeBHPD}

    def updatePriors(self, a_prior, b_prior):
        """
        Update alpha and beta priors for bayesian credible intervals

        :param a_prior: the new alpha prior
        :param b_prior: the new beta prior
        """

        # update bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # update SRS priors
        self.srs.updatePriors(a_prior, b_prior)

    def estimate(self, sample):
        """
        Estimate the KG accuracy based on the sample

        :param sample: input sample (i.e., clusters of triples) used for estimation
        :return: KG accuracy estimate
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster)/len(cluster) for cluster in sample]
        # compute estimate
        return sum(cae)/len(cae)

    def computeVar(self, sample):
        """
        Compute the sample variance

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: sample variance
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster) / len(cluster) for cluster in sample]
        # compute estimate
        ae = sum(cae) / len(cae)

        # count number of clusters in sample
        n = len(sample)

        if n*(n-1) != 0:  # compute variance
            var = (1/(n*(n-1)))*sum([(cae[i] - ae) ** 2 for i in range(n)])
        else:  # set variance to inf
            var = np.inf
        return var

    def computeESS(self, sample, numT):
        """
        Compute the Effective Sample Size as originally defined by Kish

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the effective sample size
        """

        # compute TWCS and SRS variances
        twcs_var = self.computeVar(sample)
        srs_var = self.srs.computeVar([triple for cluster in sample for triple in cluster])
        # handle corner cases
        if np.isclose(srs_var, 0) or np.isclose(twcs_var, 0):
            return numT
        # compute design effect and derive effective sample size
        d_eff = twcs_var / srs_var
        ess = numT / d_eff
        return ess

    def computeBHPD(self, sample, numT):
        """
        Compute the Bayesian Highest Posterior Density Credible Interval (CrI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        # compute effective sample size
        n = self.computeESS(sample, numT)
        if n == numT:  # effective sample size equal to actual sample size
            # compute number of successes
            x = sum([sum(c) for c in sample])
        else:  # effective sample size different from actual sample size
            # derive approximated number of successes
            x = n * ae
        # sanity check
        assert 0 <= x <= n

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if np.isclose(x, 0):  # posterior distr has mode in x = 0 -- HPD interval is (0, q(1-alpha))
            return 0, posterior.ppf(1 - self.alpha)

        if np.isclose(n-x, 0):  # posterior distr has mode in x = n -- HPD interval is (q(alpha), 1)
            return posterior.ppf(self.alpha), 1

        # compute credible mass
        mass = 1 - self.alpha

        # objective function to minimize -- i.e. interval width
        def objective(params):
            lower, upper = params
            return upper - lower

        # constraint: interval should contain credible mass
        def constraint(params):
            lower, upper = params
            return posterior.cdf(upper) - posterior.cdf(lower) - mass

        # initial guess for the interval -- rely on corresponding ET CrI (w/o ad hoc changes)
        guess = posterior.ppf([self.alpha/2, 1-self.alpha/2])

        # minimize the width of the interval (objective) subject to the credible mass (constraint)
        res = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)])

        if res.success:  # optimization succeeded -- return the derived HPD CrI
            return res.x
        else:  # optimization failed -- increase the number of performed iterations to 1000 -- default = 100
            print('optimization failed w/ standard number of iterations (100) -- increase to 1000 iterations and retry')
            rres = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)], options={'maxiter': 1000})
            if rres.success:  # optimization succeeded -- return the derived HPD CrI
                print('optimization succeeded w/ 1000 iterations')
                return rres.x
            else:  # optimization failed -- unexpected behavior, raise exception
                raise RuntimeError('optimization failed w/ 1000 iterations -- unexpected behavior, inspection required')

    def run(self, kg, groundTruth, stageTwoSize=5, minSample=30, thrMoE=0.05, ciMethod='bayesHPD', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ TWCS and stop when MoE <= thr

        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param stageTwoSize: second-stage sample size.
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute 1-alpha Intervals (CIs)
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        if self.a_prior == -1 and self.b_prior == -1 and ciMethod == 'bayesHPD':
            adaptive = True
        else:
            adaptive = False

        # prepare (head) clusters
        clusters = {}
        for id_, triple in kg:  # iterate over KG triples and make clusters
            if triple[0] in clusters:  # cluster found -- add triple (id)
                clusters[triple[0]] += [id_]
            else:  # cluster not found -- create cluster and add triple (id)
                clusters[triple[0]] = [id_]

        # get cluster heads
        heads = list(clusters.keys())
        # get cluster sizes
        sizes = [len(clusters[s]) for s in heads]
        # compute cluster weights based on cluster sizes
        weights = [sizes[i]/sum(sizes) for i in range(len(sizes))]

        estimates = []
        for _ in tqdm(range(iters)):  # perform iterations
            # set params
            lowerB = 0.0
            upperB = 1.0
            numC = 0
            numT = 0
            sample = []

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than or equal to thrMoE
                # perform TWCS over clusters
                head = random.choices(population=heads, weights=weights, k=1)[0]
                # increase heads number
                numC += 1

                # second-stage sampling
                pool = clusters[head]
                stageTwo = random.sample(pool, min(stageTwoSize, len(pool)))

                # get annotations for triples within sample
                sample += [[groundTruth[triple] for triple in stageTwo]]
                # increase triples number
                numT += len(stageTwo)

                if numT >= minSample:  # compute MoE
                    if adaptive:  # conduct adaptive strategy
                        # set temporary interval boundaries
                        lowerMin = 0.0
                        upperMin = 1.0
                        for prior in [1/3, 1/2, 1]:  # iterate over non-informative priors
                            # update prior
                            self.updatePriors(prior, prior)
                            # compute interval given prior
                            lowerCurrent, upperCurrent = self.computeMoE[ciMethod](sample, numT)
                            if upperCurrent-lowerCurrent < upperMin-lowerMin:  # computed interval has smaller width than min -- update
                                lowerMin = lowerCurrent
                                upperMin = upperCurrent
                        # set interval w/ minimal boundaries
                        lowerB = lowerMin
                        upperB = upperMin
                    else:
                        lowerB, upperB = self.computeMoE[ciMethod](sample, numT)

            # compute cost function
            cost = clusterCostFunction(numC, numT, c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[numT, estimate, cost, lowerB, upperB]]
        # return stats
        return estimates


class STWCSSampler(object):
    """
    This class represents the Stratified Two-stage Weighted Cluster Sampling (STWCS) scheme used to perform KG accuracy evaluation.
    The STWCS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05, a_prior=1, b_prior=1):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        :param a_prior: the user defined alpha prior for bayesian credible intervals
        :param b_prior: the user defined beta prior for bayesian credible intervals
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

        # bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # instantiate the SRS sampling method
        self.srs = SRSSampler(self.alpha, self.a_prior, self.b_prior)
        # instantiate the TWCS sampling method
        self.twcs = TWCSSampler(self.alpha, self.a_prior, self.b_prior)

        # possible 1-alpha intervals
        self.computeMoE = {'bayesHPD': self.computeBHPD}

    def updatePriors(self, a_prior, b_prior):
        """
        Update alpha and beta priors for bayesian credible intervals

        :param a_prior: the new alpha prior
        :param b_prior: the new beta prior
        """

        # update bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # update SRS priors
        self.srs.updatePriors(a_prior, b_prior)
        # update TWCS priors
        self.twcs.updatePriors(a_prior, b_prior)

    def estimate(self, strataSamples, strataWeights):
        """
        Estimate the KG accuracy based on the sample

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :return: KG accuracy estimate
        """

        # compute, for each stratum sample, the TWCS based accuracy estimate
        sae = [self.twcs.estimate(stratumSample) for stratumSample in strataSamples]
        # compute estimate
        return sum([sae[i] * strataWeights[i] for i in range(len(strataSamples))])

    def computeVar(self, strataSamples, strataWeights):
        """
        Compute the sample variance

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :return: sample standard deviation
        """

        # compute, for each stratum, the TWCS estimated variance
        strataVars = [self.twcs.computeVar(stratumSample) for stratumSample in strataSamples]
        # compute variance
        return sum([(strataVars[i]) * (strataWeights[i] ** 2) for i in range(len(strataSamples))])

    def computeESS(self, strataSamples, strataWeights, numT):
        """
        Compute the Effective Sample Size as originally defined by Kish

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param numT: total number of triples in the sample
        :return: the effective sample size
        """

        # compute STWCS and SRS variances
        stwcs_var = self.computeVar(strataSamples, strataWeights)
        srs_var = self.srs.computeVar([triple for sample in strataSamples for cluster in sample for triple in cluster])
        # handle corner cases
        if np.isclose(srs_var, 0) or np.isclose(stwcs_var, 0):
            return numT
        # compute design effect and derive effective sample size
        d_eff = stwcs_var / srs_var
        ess = numT / d_eff
        return ess

    def computeBHPD(self, strataSamples, strataWeights, strataT):
        """
        Compute the Bayesian Highest Posterior Density Credible Interval (CrI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)

        # get number of sample triples and clusters, as well as number of strata
        numT = sum(strataT)

        # compute effective sample size
        n = self.computeESS(strataSamples, strataWeights, numT)
        if n == numT:  # effective sample size equal to actual sample size
            # compute number of successes
            x = sum([sum(c) for sample in strataSamples for c in sample])
        else:  # effective sample size different from actual sample size
            # derive approximated number of successes
            x = n * ae
        # sanity check
        assert 0 <= x <= n

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if np.isclose(x, 0):  # posterior distr has mode in x = 0 -- HPD interval is (0, q(1-alpha))
            return 0, posterior.ppf(1 - self.alpha)

        if np.isclose(n-x, 0):  # posterior distr has mode in x = n -- HPD interval is (q(alpha), 1)
            return posterior.ppf(self.alpha), 1

        # compute credible mass
        mass = 1 - self.alpha

        # objective function to minimize -- i.e. interval width
        def objective(params):
            lower, upper = params
            return upper - lower

        # constraint: interval should contain credible mass
        def constraint(params):
            lower, upper = params
            return posterior.cdf(upper) - posterior.cdf(lower) - mass

        # initial guess for the interval -- rely on corresponding ET CrI (w/o ad hoc changes)
        guess = posterior.ppf([self.alpha/2, 1-self.alpha/2])

        # minimize the width of the interval (objective) subject to the credible mass (constraint)
        res = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)])

        if res.success:  # optimization succeeded -- return the derived HPD CrI
            return res.x
        else:  # optimization failed -- increase the number of performed iterations to 1000 -- default = 100
            print('optimization failed w/ standard number of iterations (100) -- increase to 1000 iterations and retry')
            rres = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)], options={'maxiter': 1000})
            if rres.success:  # optimization succeeded -- return the derived HPD CrI
                print('optimization succeeded w/ 1000 iterations')
                return rres.x
            else:  # optimization failed -- unexpected behavior, raise exception
                raise RuntimeError('optimization failed w/ 1000 iterations -- unexpected behavior, inspection required')

    def run(self, kg, groundTruth, preds=None, numStrata=10, stratFeature='size', targetLLM='', stageTwoSize=5, minSample=30, thrMoE=0.05, ciMethod='bayesHPD', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ STWCS and stop when MoE <= thr

        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param preds: LLM predictions on fact correctness ({LLM1: {fact1: pred, ...}, LLM2: {fact1: pred, ...}, ...})
        :param numStrata: number of considered strata
        :param stratFeature: target stratification feature. Note that "acc" represents the oracle stratification feature (used to estimate lower bound only).
        :param targetLLM: target LLM -- used to stratify w/ only one LLM (e.g., 'cohere')
        :param stageTwoSize: second-stage sample size
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute 1-alpha Intervals (CIs)
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        if self.a_prior == -1 and self.b_prior == -1 and ciMethod == 'bayesHPD':
            adaptive = True
        else:
            adaptive = False

        # prepare (head) clusters
        clusters = {}
        for id_, triple in kg:  # iterate over KG triples and make clusters
            if triple[0] in clusters:  # cluster found -- add triple (id)
                clusters[triple[0]] += [id_]
            else:  # cluster not found -- create cluster and add triple (id)
                clusters[triple[0]] = [id_]

        # get cluster heads
        heads = list(clusters.keys())
        # get cluster sizes
        sizes = [len(clusters[s]) for s in heads]

        if stratFeature == 'acc':  # compute (oracle) cluster accuracy and use it to stratify -- only use to compute lower bound
            accs = computeOracleFeature(clusters, groundTruth)
            strata = stratifyCSRF(accs, numStrata, merging=False)
        elif stratFeature == 'llm':  # compute LLM accuracy and use it to stratify
            if preds and targetLLM:
                accLLM = computeLLMFeature(preds, targetLLM, clusters)
                strata = stratifyCSRF(accLLM, numStrata, merging=True)
            elif preds and not targetLLM:
                raise Exception("The 'targetLLM' parameter must be specified when using 'llm' stratFeature")
            elif not preds and targetLLM:
                raise Exception("The 'usePreds' parameter must be set to True when using 'llm' stratFeature")
            else:
                raise Exception("The 'usePreds' and 'targetLLM' parameters must be set to True and specified, respectively, when using 'llm' stratFeature")
        elif stratFeature == 'llm-agg':  # compute LLM (aggregated) cluster score and use it to stratify
            if preds:
                aggLLM = computeLLMAggFeature(preds, clusters)
                strata = stratifyCSRF(aggLLM, numStrata, merging=True)
            else:
                raise Exception("The 'usePreds' parameter must be set to True when using 'llm-agg' stratFeature")
        elif stratFeature == 'size':  # use cluster sizes to stratify
            strata = stratifyCSRF(sizes, numStrata, merging=True)
        elif stratFeature == 'random':  # use random annotations to stratify
            rand = computeRandomFeature(clusters)
            strata = stratifyCSRF(rand, numStrata, merging=True)
        else:
            raise Exception("The provided stratification feature is not part of the considered ones: ['acc', 'llm', 'llm-agg', 'size', 'random']")

        # sanity check -- strata must contain the entire kg clusters
        assert sum([len(stratum) for stratum in strata]) == len(heads)

        if len(strata) < numStrata:  # update number of strata
            numStrata = len(strata)

        # compute strata weights
        strataWeights = [sum([sizes[i] for i in stratum])/len(kg) for stratum in strata]
        # sanity check -- strataWeights should (approximately) sum to 1
        assert np.isclose(1-sum(strataWeights), 0)

        # partition data by stratum
        headsXstratum = [[heads[i] for i in stratum] for stratum in strata]
        sizesXstratum = [[sizes[i] for i in stratum] for stratum in strata]
        weightsXstratum = [[size/sum(stratumSizes) for size in stratumSizes] for stratumSizes in sizesXstratum]

        estimates = []
        for _ in tqdm(range(iters)):  # perform iterations
            # set params
            lowerB = 0.0
            upperB = 1.0
            strataC = [0 for _ in range(numStrata)]
            strataT = [0 for _ in range(numStrata)]
            strataSamples = [[] for _ in range(numStrata)]

            warmup = list(range(numStrata)) * 2  # warmup used to ensure clusters within every stratum
            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than or equal to thrMoE
                if warmup:  # there are still strata to sample from warmup
                    ix = warmup.pop()  # pop stratum
                else:  # warmup finished -- sample strata w/ pps
                    ix = random.choices(population=range(numStrata), weights=strataWeights, k=1)[0]

                # perform TWCS over stratum ix
                head = random.choices(population=headsXstratum[ix], weights=weightsXstratum[ix], k=1)[0]
                # increase heads number
                strataC[ix] += 1

                # second-stage sampling
                pool = clusters[head]
                stageTwo = random.sample(pool, min(stageTwoSize, len(pool)))

                # get annotations for triples within sample
                strataSamples[ix] += [[groundTruth[triple] for triple in stageTwo]]
                # increase triples number
                strataT[ix] += len(stageTwo)

                if sum(strataT) >= minSample and not warmup:  # compute CI if enough triples in sample and all strata sampled at least twice
                    if adaptive:  # conduct adaptive strategy
                        # set temporary interval boundaries
                        lowerMin = 0.0
                        upperMin = 1.0
                        for prior in [1/3, 1/2, 1]:  # iterate over non-informative priors
                            # update prior
                            self.updatePriors(prior, prior)
                            # compute interval given prior
                            lowerCurrent, upperCurrent = self.computeMoE[ciMethod](strataSamples, strataWeights, strataT)
                            if upperCurrent-lowerCurrent < upperMin-lowerMin:  # computed interval has smaller width than min -- update
                                lowerMin = lowerCurrent
                                upperMin = upperCurrent
                        # set interval w/ minimal boundaries
                        lowerB = lowerMin
                        upperB = upperMin
                    else:
                        lowerB, upperB = self.computeMoE[ciMethod](strataSamples, strataWeights, strataT)

            # compute cost function
            cost = clusterCostFunction(sum(strataC), sum(strataT), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(strataSamples, strataWeights)

            # store stats
            estimates += [[sum(strataT), estimate, cost, lowerB, upperB]]
        # return stats
        return estimates
