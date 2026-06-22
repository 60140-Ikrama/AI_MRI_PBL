import numpy as np
import scipy.stats as stats

# =====================================================================
# 1. Hypothesis Testing
# =====================================================================

def run_independent_ttest(group_a, group_b):
    """
    Computes an Independent two-sample t-test.
    """
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    ci_a = stats.t.interval(0.95, len(group_a)-1, loc=np.mean(group_a), scale=stats.sem(group_a))
    ci_b = stats.t.interval(0.95, len(group_b)-1, loc=np.mean(group_b), scale=stats.sem(group_b))
    
    return {
        "Test": "Independent t-test",
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "Group_A_Mean": float(np.mean(group_a)),
        "Group_B_Mean": float(np.mean(group_b)),
        "Group_A_CI": (float(ci_a[0]), float(ci_a[1])),
        "Group_B_CI": (float(ci_b[0]), float(ci_b[1])),
        "Significant": bool(p_val < 0.05)
    }

def run_paired_ttest(group_a, group_b):
    """
    Computes a Paired two-sample t-test (e.g., same patient, different models).
    """
    t_stat, p_val = stats.ttest_rel(group_a, group_b)
    diff = np.array(group_a) - np.array(group_b)
    ci_diff = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    
    return {
        "Test": "Paired t-test",
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "Mean_Difference": float(np.mean(diff)),
        "Difference_CI": (float(ci_diff[0]), float(ci_diff[1])),
        "Significant": bool(p_val < 0.05)
    }

def run_one_way_anova(*groups):
    """
    Computes a One-Way ANOVA across multiple groups.
    """
    f_stat, p_val = stats.f_oneway(*groups)
    means = [float(np.mean(g)) for g in groups]
    sems = [float(stats.sem(g)) for g in groups]
    
    return {
        "Test": "One-Way ANOVA",
        "F_statistic": float(f_stat),
        "p_value": float(p_val),
        "Means": means,
        "SEMs": sems,
        "Significant": bool(p_val < 0.05)
    }

def run_repeated_measures_anova(data_matrix):
    """
    Computes a Repeated Measures ANOVA.
    data_matrix: (N patients, K models)
    """
    n, k = data_matrix.shape
    
    # Calculate Sum of Squares
    grand_mean = np.mean(data_matrix)
    ss_total = np.sum((data_matrix - grand_mean) ** 2)
    
    patient_means = np.mean(data_matrix, axis=1)
    ss_subjects = k * np.sum((patient_means - grand_mean) ** 2)
    
    model_means = np.mean(data_matrix, axis=0)
    ss_models = n * np.sum((model_means - grand_mean) ** 2)
    
    ss_error = ss_total - ss_subjects - ss_models
    
    df_models = k - 1
    df_error = (n - 1) * (k - 1)
    
    ms_models = ss_models / df_models
    ms_error = ss_error / df_error
    
    f_stat = ms_models / ms_error
    p_val = 1 - stats.f.cdf(f_stat, df_models, df_error)
    
    return {
        "Test": "Repeated Measures ANOVA",
        "F_statistic": float(f_stat),
        "p_value": float(p_val),
        "df_models": int(df_models),
        "df_error": int(df_error),
        "Significant": bool(p_val < 0.05)
    }

def run_mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Computes McNemar's test for paired binary classifications.
    y_true: true binary labels (0/1)
    y_pred1: prediction of model 1 (0/1)
    y_pred2: prediction of model 2 (0/1)
    """
    # Create contingency table
    # b: model 1 correct, model 2 incorrect
    # c: model 1 incorrect, model 2 correct
    
    b = 0
    c = 0
    for t, p1, p2 in zip(y_true, y_pred1, y_pred2):
        correct1 = (p1 == t)
        correct2 = (p2 == t)
        if correct1 and not correct2:
            b += 1
        elif not correct1 and correct2:
            c += 1
            
    if b + c == 0:
        p_val = 1.0
        chi2_stat = 0.0
    else:
        # Continuity correction
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_val = stats.chi2.sf(chi2_stat, 1)
        
    return {
        "Test": "McNemar Test",
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_val),
        "Contingency_b": int(b),
        "Contingency_c": int(c),
        "Significant": bool(p_val < 0.05)
    }

# =====================================================================
# 2. Effect Sizes & Resampling
# =====================================================================

def compute_cohens_d(group_a, group_b):
    """
    Computes Cohen's d effect size for two independent groups.
    """
    n1, n2 = len(group_a), len(group_b)
    v1, v2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    
    if s_pooled == 0:
        return 0.0
        
    d = (np.mean(group_a) - np.mean(group_b)) / s_pooled
    return float(d)

def run_bootstrap_validation(data, num_bootstraps=1000, ci_level=0.95):
    """
    Performs bootstrap resampling to estimate empirical mean and CI.
    """
    boot_means = []
    n = len(data)
    for _ in range(num_bootstraps):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
        
    boot_means = np.array(boot_means)
    mean_val = np.mean(boot_means)
    
    # Percentile CI
    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100
    
    ci_lower = np.percentile(boot_means, lower_pct)
    ci_upper = np.percentile(boot_means, upper_pct)
    
    return {
        "Bootstrap_Mean": float(mean_val),
        "Empirical_CI": (float(ci_lower), float(ci_upper)),
        "Resampling_Count": num_bootstraps
    }
