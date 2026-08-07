import os
from experiment_utils import GPUJobScheduler, check_patient_leakage

def get_patients_for_fold(fold: int, num_folds: int, all_patients: list, strategy: str):
    # Dummy mock implementation to show logic
    if strategy == "LOCO":
        test_patients = [all_patients[fold]]
        train_patients = [p for p in all_patients if p not in test_patients]
        val_patients = train_patients[:len(train_patients)//10]
        train_patients = [p for p in train_patients if p not in val_patients]
    else:
        fold_size = len(all_patients) // num_folds
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < num_folds - 1 else len(all_patients)
        
        test_patients = all_patients[test_start:test_end]
        remaining = [p for p in all_patients if p not in test_patients]
        
        val_size = len(remaining) // (num_folds - 1) if num_folds > 1 else len(remaining)//10
        val_patients = remaining[:val_size]
        train_patients = remaining[val_size:]
        
    return train_patients, val_patients, test_patients

def main():
    manifest_path = "run_cv_manifest.json"
    scheduler = GPUJobScheduler(manifest_path=manifest_path)
    
    cv_strategies = {
        "5-fold": 5,
        "10-fold": 10,
        "LOCO": 15  # Assume 15 cohorts/centers/patients
    }
    
    all_patients = [f"patient_{i}" for i in range(100)]
    
    for strategy_name, num_folds in cv_strategies.items():
        for fold in range(num_folds):
            train_p, val_p, test_p = get_patients_for_fold(fold, num_folds, all_patients, strategy_name)
            
            # Strictly check patient leakage
            check_patient_leakage(train_p, val_p, test_p)
            
            job_id = f"cv_{strategy_name}_fold_{fold}"
            command = f"python train.py --cv_strategy {strategy_name} --fold {fold}"
            scheduler.add_job(job_id, command)
            
    scheduler.run()

if __name__ == "__main__":
    main()
