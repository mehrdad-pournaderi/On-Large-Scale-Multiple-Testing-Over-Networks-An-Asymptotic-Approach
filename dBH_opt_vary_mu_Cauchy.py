import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
import math

# -----------------------------
# Helper functions
# -----------------------------

def generate_pvalues_cauchy(n, k, mu):
    """
    Generate p-values under a Cauchy alternative model.
    
    Parameters
    ----------
    n : int
        Total number of tests
    k : int
        Number of alternatives
    mu : float
        Location shift for alternatives
    
    Returns
    -------
    p_values : np.ndarray
        Array of p-values (size n)
    """
    gam = 1.0

    # Alternatives
    alts = mu + (np.random.rand(k) - 0.5) + gam * np.tan(np.pi * (np.random.rand(k) - 0.5))

    # Nulls
    nulls = gam * np.tan(np.pi * (np.random.rand(n - k) - 0.5))

    tests = np.concatenate([alts, nulls])

    # Compute one-sided p-values
    p_values = 1 - (1 / np.pi * np.arctan(tests / gam) + 0.5)

    return p_values


def generate_pvalues_composite(n, k, mu):
    """
    Generate p-values under the composite Gaussian alternative.
    
    Parameters
    ----------
    n : int
        Total number of tests
    k : int
        Number of alternative hypotheses
    mu : float
        Mean shift for alternatives
    
    Returns
    -------
    p_values : np.ndarray
        Array of p-values (size n)
    """
    ss = 1  # standard deviation
    tests = np.zeros(n)

    # Alternatives
    for i in range(k):
        sign = 1  # MATLAB had random sign commented out
        tests[i] = np.random.normal(mu * sign + (np.random.rand() - 0.5), ss)

    # Nulls
    tests[k:n] = np.random.normal(0, ss, size=n-k)

    # One-sided p-values
    p_values = 1 - norm.cdf(tests, loc=0, scale=ss)
    
    return p_values


def estimate_nulls_spacing(p_values, n):
    """
    Estimate proportion of null hypotheses using spacing method.
    
    Parameters
    ----------
    p_values : array-like
        List/array of p-values
    n : int
        Total number of tests
    
    Returns
    -------
    ratio_null_hat : float
        Estimate of null proportion
    """
    d = 0.5
    # MATLAB used n^(3.5/5); original commented: n^(4/5)*(log(n))^(-2d)
    r = int(np.floor(n ** (3.5 / 5)))
    
    Y = np.sort(p_values)
    temp = []
    
    for k in range(r, n - r):  # MATLAB indexing (1+r : n-r) -> Python (r : n-r)
        temp.append(Y[k + r] - Y[k - r])
    
    if len(temp) == 0:
        return 0.0
    
    V = max(temp)
    
    if V == 0:
        return 0.0
    
    ratio_null_hat = min(1.0, 2 * r / (n * V))
    
    return ratio_null_hat



def BH(pvalues, alpha, n=None):
    """
    Benjamini-Hochberg procedure.
    Returns (rejected_indices, k_rejected)
    """
    if n is None:
        n = len(pvalues)
    p = np.asarray(pvalues)
    order = np.argsort(p)
    p_sorted = p[order]
    thresh = alpha * (np.arange(1, n+1) / n)
    below = p_sorted <= thresh
    if np.any(below):
        k = int(np.max(np.where(below)[0]) + 1)
        rejected = order[:k]
    else:
        k = 0
        rejected = np.array([], dtype=int)
    return rejected, k


def compute_fdr_power_from_rejected_indices(rejected_indices, tot_alt):
    """
    Given indices (0-based) of rejected items in a pooled vector where
    the first `tot_alt` entries correspond to true alternatives, compute FDR and power.
    """
    if len(rejected_indices) == 0:
        return 0.0, 0.0
    rejected = np.asarray(rejected_indices)
    wrong = np.sum(rejected >= tot_alt)
    correct = np.sum(rejected < tot_alt)
    fdr = wrong / len(rejected)
    power = correct / tot_alt if tot_alt > 0 else 0.0
    return float(fdr), float(power)

# -----------------------------
# Simulation including Segmentation (Greedy) method
# -----------------------------

def run_simulation(n_experiment=100, n_nodes=5, alpha=0.2, n=1000, seed=0):
    rng = np.random.default_rng(seed)

    mu_range = np.arange(0.5, 10.01, 0.25)

    final_FDR_pool, final_power_pool = [], []
    final_FDR_prop, final_power_prop = [], []
    final_FDR_noco, final_power_noco = [], []
    final_FDR_seg, final_power_seg = [], []

    for c in mu_range:
        # node sizes (mirrors MATLAB: flip of ceil(linspace(0.2,1,n_nodes)*n))
        node = np.flip(np.ceil(np.linspace(0.2, 1, n_nodes) * n).astype(int))
        a = np.flip(np.linspace(0.1, 0.5, n_nodes))
        k = np.floor(a * node).astype(int)
        tot_alt = np.sum(k)

        FDR_pool_list, power_pool_list = [], []
        FDR_prop_list, power_prop_list = [], []
        FDR_noco_list, power_noco_list = [], []
        FDR_seg_list, power_seg_list = [], []

        mu = c * np.ones(n_nodes)

        for i in range(n_experiment):
            # ----- generate p-values for each node -----
            pvalues = []
            n0 = np.zeros(n_nodes)
            for j in range(n_nodes):
                pv = generate_pvalues_cauchy(node[j], k[j], mu[j])
                # Shuffle pv? MATLAB places alternatives first; keep same ordering
                pvalues.append(pv)
                n0[j] = estimate_nulls_spacing(pv[:node[j]], node[j])

            # weighted global null proportion
            a0_hat = float((node * n0).sum() / node.sum())

            # ---------------- pooled BH -----------------
            # Create pooled p-value vector in the same layout as MATLAB
            pooled_alts = np.concatenate([pvalues[j][:k[j]] for j in range(n_nodes)]) if tot_alt > 0 else np.array([])
            pooled_nulls = np.concatenate([pvalues[j][k[j]:node[j]] for j in range(n_nodes)])
            pool_pvalue = np.concatenate([pooled_alts, pooled_nulls])

            # run BH with adjusted level (as in MATLAB: alpha / a0_hat)
            adj_alpha_pool = alpha / (a0_hat + 1e-12)
            rejected_pool, pool_num_rej = BH(pool_pvalue, adj_alpha_pool, len(pool_pvalue))
            F_pool, P_pool = compute_fdr_power_from_rejected_indices(rejected_pool, tot_alt)
            FDR_pool_list.append(F_pool)
            power_pool_list.append(P_pool)

            # --------------- Proportion Aggregation ---------------
            beta_g = (a0_hat / alpha - a0_hat) / (1 - a0_hat + 1e-12)
            num_rej = np.zeros(n_nodes, dtype=int)
            wrong_rej = np.zeros(n_nodes, dtype=int)
            correct_rej = np.zeros(n_nodes, dtype=int)
            for j in range(n_nodes):
                denom = (1 - n0[j]) * beta_g + n0[j]
                loc_alpha = 1.0 / (denom + 1e-12)
                rejected_j, num = BH(pvalues[j][:node[j]], loc_alpha, node[j])
                num_rej[j] = num
                # Because alternatives are placed first in pvalues[j], indices < k[j] are true alts
                wrong_rej[j] = np.sum(rejected_j >= k[j]) if num > 0 else 0
                correct_rej[j] = np.sum(rejected_j < k[j]) if num > 0 else 0
            tot_rej = int(num_rej.sum())
            tot_wrong = int(wrong_rej.sum())
            tot_correct = int(correct_rej.sum())
            FDR_prop_list.append(tot_wrong / max(tot_rej, 1))
            power_prop_list.append(tot_correct / tot_alt if tot_alt > 0 else 0.0)

            # ----------------- No-communication ------------------
            num_rej_noco = np.zeros(n_nodes, dtype=int)
            wrong_rej_noco = np.zeros(n_nodes, dtype=int)
            correct_rej_noco = np.zeros(n_nodes, dtype=int)
            for j in range(n_nodes):
                # MATLAB used alpha / n0(j)
                loc_alpha_noco = alpha / (n0[j] + 1e-12)
                rej_j, num = BH(pvalues[j][:node[j]], loc_alpha_noco, node[j])
                num_rej_noco[j] = num
                wrong_rej_noco[j] = np.sum(rej_j >= k[j]) if num > 0 else 0
                correct_rej_noco[j] = np.sum(rej_j < k[j]) if num > 0 else 0
            tot_rej_noco = int(num_rej_noco.sum())
            tot_wrong_noco = int(wrong_rej_noco.sum())
            tot_correct_noco = int(correct_rej_noco.sum())
            FDR_noco_list.append(tot_wrong_noco / max(tot_rej_noco, 1))
            power_noco_list.append(tot_correct_noco / tot_alt if tot_alt > 0 else 0.0)

            # ----------------- Segmentation (Greedy agg) ------------------
            # compute eps, step sizes, and histogram matrix O
            eps = 2.5 * alpha / np.sqrt(node.sum())
            denom_stp = (n0 * node) / node.sum()
            # avoid division by zero
            denom_stp[denom_stp == 0] = 1e-12
            stp = eps / denom_stp
            total_num_bins = int(np.sum(np.floor(1 / stp)))
            # print("Total number of bins (approx):", total_num_bins)
            # construct histogram matrix O
            min_stp = np.min(stp)
            L = int(np.ceil(1.0 / (min_stp + 1e-12)))
            if L < 1:
                L = 1
            O = np.zeros((n_nodes, L))
            for j in range(n_nodes):
                for l0 in range(L):
                    lower = l0 * stp[j]
                    upper = (l0 + 1) * stp[j]
                    mask = (pvalues[j][:node[j]] >= lower) & (pvalues[j][:node[j]] < upper)
                    O[j, l0] = mask.sum()

            # greedy selection of histogram cells (same logic as MATLAB)
            R = 0.0
            z = 0
            # we will mutate O in-place by setting selected cells to -1
            while True:
                O_max = np.max(O)
                if O_max <= 0:
                    break
                mn = (z + 1) * (eps * node.sum())
                # lg = np.log(float(math.comb(total_num_bins, z + 1))/0.1)
                # FDR_hat = (mn + np.sqrt(2 * mn * lg)+ 2/3 * lg) / (R + O_max + 1e-12)
                FDR_hat = mn / (R + O_max + 1e-12)
                if FDR_hat <= alpha:
                    z += 1
                    R += O_max
                    # find first occurrence of O_max
                    pos = np.argwhere(O == O_max)
                    if pos.size == 0:
                        break
                    row, col = pos[0]
                    O[row, col] = -1.0
                else:
                    break

            # Collect rejections from selected cells
            I_Rej = [[] for _ in range(n_nodes)]
            for j in range(n_nodes):
                for l0 in range(L):
                    if O[j, l0] == -1.0:
                        lower = l0 * stp[j]
                        upper = (l0 + 1) * stp[j]
                        idxs = np.where((pvalues[j][:node[j]] >= lower) & (pvalues[j][:node[j]] < upper))[0]
                        # place indices in the rejection list in the order found
                        I_Rej[j].extend(idxs.tolist())

            num_rej_seg = np.array([len(x) for x in I_Rej], dtype=int)
            wrong_rej_seg = np.zeros(n_nodes, dtype=int)
            correct_rej_seg = np.zeros(n_nodes, dtype=int)
            for j in range(n_nodes):
                if num_rej_seg[j] > 0:
                    idxs = np.array(I_Rej[j])
                    wrong_rej_seg[j] = np.sum(idxs >= k[j])
                    correct_rej_seg[j] = np.sum(idxs < k[j])
            tot_rej_seg = int(num_rej_seg.sum())
            tot_wrong_seg = int(wrong_rej_seg.sum())
            tot_correct_seg = int(correct_rej_seg.sum())
            FDR_seg_list.append(tot_wrong_seg / max(tot_rej_seg, 1))
            power_seg_list.append(tot_correct_seg / tot_alt if tot_alt > 0 else 0.0)

        # store averages for this c
        final_FDR_pool.append(np.mean(FDR_pool_list))
        final_power_pool.append(np.mean(power_pool_list))
        final_FDR_prop.append(np.mean(FDR_prop_list))
        final_power_prop.append(np.mean(power_prop_list))
        final_FDR_noco.append(np.mean(FDR_noco_list))
        final_power_noco.append(np.mean(power_noco_list))
        final_FDR_seg.append(np.mean(FDR_seg_list))
        final_power_seg.append(np.mean(power_seg_list))

    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    axs[0].plot(mu_range, final_FDR_noco, '-s', label='No-com BH')
    axs[0].plot(mu_range, final_FDR_pool, '-d', label='Pooled BH')
    axs[0].plot(mu_range, final_FDR_prop, '-*', label='Prop matching')
    axs[0].plot(mu_range, final_FDR_seg, '-o', label='Greedy agg')
    axs[0].axhline(alpha, ls='--', label='Target FDR')
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('FDR')
    axs[0].legend()

    axs[1].plot(mu_range, final_power_noco, '-s', label='No-com BH')
    axs[1].plot(mu_range, final_power_pool, '-d', label='Pooled BH')
    axs[1].plot(mu_range, final_power_prop, '-*', label='Prop matching')
    axs[1].plot(mu_range, final_power_seg, '-o', label='Greedy agg')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('η')
    axs[1].set_ylabel('Power')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_simulation(n_experiment=100, n_nodes=5, alpha=0.2, n=1000, seed=42)
