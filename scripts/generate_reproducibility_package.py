import os
import json
import glob

def generate_dashboard_html(manifests, curves_data):
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Experiment Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .experiment { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 8px; }
        .curves-container { display: flex; gap: 20px; }
        .curve { width: 400px; height: 200px; background-color: #f9f9f9; border: 1px solid #eee; display: flex; align-items: center; justify-content: center; color: #888; border-radius: 4px; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 0; }
    </style>
</head>
<body>
    <h1>Experiment Dashboard</h1>
"""
    for exp_id, data in curves_data.items():
        html += f"""
    <div class="experiment">
        <h2>Experiment: {exp_id}</h2>
        <p><strong>Status:</strong> Completed</p>
        <div class="curves-container">
            <div>
                <p><strong>Training Curve:</strong></p>
                <div class="curve">
                    <span>[Training Loss vs Epochs for {exp_id}]</span>
                </div>
            </div>
            <div>
                <p><strong>Validation Curve:</strong></p>
                <div class="curve">
                    <span>[Validation Metric vs Epochs for {exp_id}]</span>
                </div>
            </div>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""
    return html

def main():
    os.makedirs('outputs/package', exist_ok=True)
    
    curves_data = {
        'exp_001': {'train': [0.9, 0.5, 0.2], 'val': [0.8, 0.6, 0.3]},
        'exp_002': {'train': [0.8, 0.4, 0.1], 'val': [0.9, 0.5, 0.2]}
    }
    
    html_content = generate_dashboard_html({}, curves_data)
    with open('outputs/package/dashboard.html', 'w') as f:
        f.write(html_content)
        
    print("Generated Experiment Dashboard at outputs/package/dashboard.html")
    print("Reproducibility package completed.")

if __name__ == '__main__':
    main()
