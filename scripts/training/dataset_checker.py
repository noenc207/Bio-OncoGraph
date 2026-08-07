import os
import json
import argparse
import csv
from collections import defaultdict

def check_corrupt(file_path):
    if not os.path.exists(file_path):
        return True
    if os.path.getsize(file_path) == 0:
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='dataset_manifest.csv', help="CSV with columns: patient_id, slide_id, slide_path")
    parser.add_argument('--output', type=str, default='dataset_summary.json')
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        with open(args.output, 'w') as f:
            json.dump({"error": f"Manifest not found: {args.manifest}"}, f)
        return

    patients = set()
    slides = set()
    duplicates = []
    missing_samples = []
    corrupt_samples = []
    
    patient_to_slides = defaultdict(list)

    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row.get('patient_id')
            slide_id = row.get('slide_id')
            slide_path = row.get('slide_path')

            if not patient_id or not slide_id or not slide_path:
                continue
            
            if slide_id in slides:
                duplicates.append(slide_id)
            else:
                slides.add(slide_id)
                patients.add(patient_id)
                patient_to_slides[patient_id].append(slide_id)
            
            if not os.path.exists(slide_path):
                missing_samples.append(slide_path)
            elif check_corrupt(slide_path):
                corrupt_samples.append(slide_path)

    summary = {
        "patient_count": len(patients),
        "slide_count": len(slides),
        "missing_samples_count": len(missing_samples),
        "duplicate_samples_count": len(duplicates),
        "corrupt_samples_count": len(corrupt_samples),
        "missing_samples": missing_samples,
        "duplicate_samples": duplicates,
        "corrupt_samples": corrupt_samples
    }

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Dataset check completed. Summary saved to {args.output}")

if __name__ == "__main__":
    main()
