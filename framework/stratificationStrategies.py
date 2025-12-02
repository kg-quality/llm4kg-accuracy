import numpy as np
from copy import deepcopy


def stratifyCSRF(clusterFeature, numStrata=10, merging=True):
    """
    Perform stratification w/ Cumulative Square Root of Frequency (CSRF) based on cluster (stratification) feature,
    and merge strata w/ near-zero variance to prevent oversplitting due to proxy signals.

    :param clusterFeature: target stratification feature (must represent the entire cluster population)
    :param numStrata: number of desired strata
    :param merging: whether to merge strata with near-zero variance after stratification
    :return: feature indices divided into strata
    """

    # step 1: CSRF basic stratification
    unique, counts = np.unique(clusterFeature, return_counts=True)
    sqrt_counts = np.sqrt(counts)
    csrf = np.cumsum(sqrt_counts)
    csrf2unique = dict(zip(csrf, unique))

    strataSize = csrf[-1] / numStrata
    boundaries = [-1] + [strataSize * (i + 1) for i in range(numStrata - 1)]

    strata = []
    for b in range(numStrata):
        if b == numStrata - 1:
            intervals = [csrf2unique[c] for c in csrf if c > boundaries[b]]
        else:
            intervals = [csrf2unique[c] for c in csrf if (c > boundaries[b]) and (c <= boundaries[b + 1])]
        stratum = [i for i in range(len(clusterFeature)) if clusterFeature[i] in intervals]
        if stratum:
            strata.append(stratum)

    # sanity check
    assert len(clusterFeature) == sum(len(stratum) for stratum in strata)
    print(f'Desired strata: {numStrata}\nStrata after CSRF: {len(strata)}')

    if merging:  # step 2: merge strata w/ intra-stratum variance close to zero
        merged = mergeStrata(strata, clusterFeature)
        print(f'Strata after merging: {len(merged)}')
        # return strata after merging
        return merged
    else:  # return strata obtained w/ CSRF
        return strata


def mergeStrata(strata, clusterFeature):
    """
    Merge strata with intra-stratum variance close to zero with the nearest neighbor strata.

    :param strata: strata obtained from stratifyCSRF
    :param clusterFeature: target stratification feature
    :return: strata w/ near-zero intra-stratum variance merged w/ nn strata
    """

    # init vars
    working = deepcopy(strata)
    epsilon = 1e-3
    changed = True

    while changed:  # keep merging until near-zero variance strata are not found anymore
        changed = False
        for i, stratum in enumerate(working):
            if len(stratum) <= 1:
                continue  # singletons we can skip for now
            values = getStratumValues(clusterFeature, stratum)
            if np.var(values) < epsilon:  # found stratum w/ near-zero variance
                # find the best stratum to merge into (based on smallest mean difference)
                best_j = None
                best_distance = float('inf')
                mean_i = np.mean(values)
                for j, other in enumerate(working):
                    if j == i or not other:
                        continue
                    mean_j = np.mean(getStratumValues(clusterFeature, other))
                    distance = abs(mean_i - mean_j)
                    if distance < best_distance:
                        best_distance = distance
                        best_j = j
                if best_j is not None:
                    working[best_j].extend(stratum)
                    working[i] = []
                    changed = True
                    break  # need to restart the loop after modifying
        working = [s for s in working if s]  # remove empty strata
    return working


def getStratumValues(clusterFeature, stratum):
    """
    Return stratum (cluster) features

    :param clusterFeature: target stratification feature (must represent the entire cluster population)
    :param stratum: the target stratum
    :return: stratum (features)
    """

    return [clusterFeature[i] for i in stratum]
