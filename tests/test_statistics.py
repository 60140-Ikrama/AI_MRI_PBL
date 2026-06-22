import numpy as np
from src.statistics import (
    run_independent_ttest, run_paired_ttest, run_one_way_anova,
    run_repeated_measures_anova, run_mcnemar_test, compute_cohens_d,
    run_bootstrap_validation
)

def test_t_tests_and_anova():
    np.random.seed(42)
    group_a = np.random.normal(0.80, 0.05, 30)
    group_b = np.random.normal(0.85, 0.05, 30)
    group_c = np.random.normal(0.90, 0.04, 30)
    
    ind_t = run_independent_ttest(group_a, group_b)
    paired_t = run_paired_ttest(group_a, group_b)
    anova = run_one_way_anova(group_a, group_b, group_c)
    
    assert ind_t["Test"] == "Independent t-test"
    assert paired_t["Test"] == "Paired t-test"
    assert anova["Test"] == "One-Way ANOVA"
    
    assert "p_value" in ind_t
    assert "t_statistic" in paired_t
    assert "F_statistic" in anova
    assert len(anova["Means"]) == 3

def test_repeated_measures_anova():
    # 30 patients, 3 models
    np.random.seed(42)
    data = np.random.normal(0.85, 0.05, (30, 3))
    res = run_repeated_measures_anova(data)
    
    assert res["Test"] == "Repeated Measures ANOVA"
    assert "F_statistic" in res
    assert "p_value" in res
    assert res["df_models"] == 2
    assert res["df_error"] == 58

def test_mcnemar_test():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    y_pred2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    # y_pred1 has 8 correct, y_pred2 has 10 correct
    # Contingency differences:
    # y_pred2 correct, y_pred1 incorrect: b=2 (index 4, 9)
    # y_pred1 correct, y_pred2 incorrect: c=0
    
    res = run_mcnemar_test(y_true, y_pred1, y_pred2)
    
    assert res["Test"] == "McNemar Test"
    assert res["Contingency_b"] == 2
    assert res["Contingency_c"] == 0
    assert "p_value" in res

def test_cohens_d_and_bootstrap():
    np.random.seed(42)
    g_a = np.random.normal(0.82, 0.05, 30)
    g_b = np.random.normal(0.88, 0.05, 30)
    
    d = compute_cohens_d(g_b, g_a)
    boot = run_bootstrap_validation(g_b, num_bootstraps=100)
    
    assert d > 0.0
    assert "Bootstrap_Mean" in boot
    assert "Empirical_CI" in boot
    assert len(boot["Empirical_CI"]) == 2
    assert boot["Empirical_CI"][0] < boot["Empirical_CI"][1]
