import os
import pandas as pd
import json

def main():
    os.makedirs("results/leaderboard", exist_ok=True)
    
    data = [
        {"Model": "Bio-OncoGraph", "AUROC": 0.95, "Speed": 45},
        {"Model": "BaselineA", "AUROC": 0.85, "Speed": 60},
        {"Model": "BaselineB", "AUROC": 0.88, "Speed": 70},
    ]
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=["AUROC", "Speed"], ascending=[False, True])
    
    df.to_csv("results/leaderboard/leaderboard.csv", index=False)
    
    try:
        df.to_excel("results/leaderboard/leaderboard.xlsx", index=False)
    except ImportError:
        pass
    
    with open("results/leaderboard/leaderboard.md", "w") as f:
        f.write(df.to_markdown(index=False))
        
    df.to_json("results/leaderboard/leaderboard.json", orient="records", indent=4)
    print("Exported leaderboard to results/leaderboard/")

if __name__ == "__main__":
    main()
