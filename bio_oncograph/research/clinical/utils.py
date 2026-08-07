"""Utility functions for clinical modeling."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def concordance_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Calculate Harrell's C-index.
    
    Args:
        risk_scores: Array of predicted risk scores (higher means higher risk).
        times: Array of survival times.
        events: Array of event indicators (1=event, 0=censored).
        
    Returns:
        The concordance index.
    """
    logger.debug("Calculating concordance index.")
    
    n = len(times)
    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0
    
    # Pure numpy implementation for C-index
    for i in range(n):
        if events[i] == 1:
            for j in range(n):
                if times[i] < times[j]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 1
                        
    total_comparisons = concordant + discordant + tied_risk
    if total_comparisons == 0:
        logger.warning("No comparable pairs found for C-index calculation.")
        return 0.5
        
    c_index = (concordant + 0.5 * tied_risk) / total_comparisons
    return c_index


def log_rank_test(times_group1: np.ndarray, events_group1: np.ndarray, 
                  times_group2: np.ndarray, events_group2: np.ndarray) -> float:
    """Calculate the log-rank test chi-squared statistic.
    
    Args:
        times_group1: Survival times for group 1.
        events_group1: Event indicators for group 1.
        times_group2: Survival times for group 2.
        events_group2: Event indicators for group 2.
        
    Returns:
        The chi-squared statistic.
    """
    logger.debug("Calculating log-rank test.")
    
    times = np.concatenate([times_group1, times_group2])
    events = np.concatenate([events_group1, events_group2])
    groups = np.concatenate([np.zeros(len(times_group1)), np.ones(len(times_group2))])
    
    unique_times = np.unique(times[events == 1])
    
    o1, o2 = 0.0, 0.0
    e1, e2 = 0.0, 0.0
    
    for t in unique_times:
        n1 = np.sum(times_group1 >= t)
        n2 = np.sum(times_group2 >= t)
        n = n1 + n2
        
        if n == 0:
            continue
            
        d1 = np.sum((times_group1 == t) & (events_group1 == 1))
        d2 = np.sum((times_group2 == t) & (events_group2 == 1))
        d = d1 + d2
        
        o1 += d1
        o2 += d2
        e1 += (n1 / n) * d
        e2 += (n2 / n) * d
        
    if e1 == 0 or e2 == 0:
        return 0.0
        
    chi2 = ((o1 - e1)**2) / e1 + ((o2 - e2)**2) / e2
    return chi2


def integrated_brier_score(predicted_surv: np.ndarray, times: np.ndarray, 
                           events: np.ndarray, horizons: np.ndarray) -> float:
    """Calculate the Integrated Brier Score.
    
    Args:
        predicted_surv: Matrix of predicted survival probabilities (samples x horizons).
        times: Array of survival times.
        events: Array of event indicators.
        horizons: Array of time horizons corresponding to columns in predicted_surv.
        
    Returns:
        The integrated Brier score.
    """
    logger.debug("Calculating integrated Brier score.")
    
    num_horizons = len(horizons)
    brier_scores = np.zeros(num_horizons)
    
    for i, t_star in enumerate(horizons):
        # Simplified Brier score without IPCW for brevity, as per pure numpy constraints
        actual_surv = (times > t_star).astype(float)
        valid_mask = (times > t_star) | (events == 1)
        
        if np.sum(valid_mask) > 0:
            brier_scores[i] = np.mean((actual_surv[valid_mask] - predicted_surv[valid_mask, i])**2)
            
    if num_horizons > 1:
        dt = np.diff(horizons)
        ibs = np.sum(dt * (brier_scores[:-1] + brier_scores[1:]) / 2) / (horizons[-1] - horizons[0])
        return float(ibs)
    elif num_horizons == 1:
        return float(brier_scores[0])
    return 0.0


def time_dependent_auc(risk_scores: np.ndarray, times: np.ndarray, 
                       events: np.ndarray, horizon: float) -> float:
    """Calculate a simple time-dependent AUC at a specific horizon.
    
    Args:
        risk_scores: Array of risk scores.
        times: Array of survival times.
        events: Array of event indicators.
        horizon: The time horizon.
        
    Returns:
        The time-dependent AUC.
    """
    logger.debug("Calculating time-dependent AUC at horizon %f.", horizon)
    
    # Define cases (events before horizon) and controls (survival beyond horizon)
    is_case = (times <= horizon) & (events == 1)
    is_control = (times > horizon)
    
    cases_risk = risk_scores[is_case]
    controls_risk = risk_scores[is_control]
    
    if len(cases_risk) == 0 or len(controls_risk) == 0:
        logger.warning("No cases or controls found for time-dependent AUC at horizon %f.", horizon)
        return 0.5
        
    # Calculate AUC
    concordant = 0
    ties = 0
    
    for cr in cases_risk:
        concordant += np.sum(cr > controls_risk)
        ties += np.sum(cr == controls_risk)
        
    total_pairs = len(cases_risk) * len(controls_risk)
    auc = (concordant + 0.5 * ties) / total_pairs
    return float(auc)
