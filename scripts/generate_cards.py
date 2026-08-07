import os

def generate_cards():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cards_dir = os.path.join(base_dir, 'cards')
    os.makedirs(cards_dir, exist_ok=True)

    model_card = """# Model Card for Bio-OncoGraph Model A

## Model Details
* **Name**: Bio-OncoGraph Model A
* **Version**: 1.0.0
* **Type**: Graph Neural Network

## Intended Use
* **Primary Intended Uses**: Oncology research and prediction.
* **Out-of-Scope Use Cases**: Direct clinical diagnosis.
"""
    with open(os.path.join(cards_dir, 'model_card.md'), 'w') as f:
        f.write(model_card)

    dataset_card = """# Dataset Card for Bio-OncoGraph Dataset A

## Dataset Details
* **Name**: Bio-OncoGraph Dataset A
* **Version**: 1.0.0
* **Format**: Parquet/CSV

## Dataset Creation
* **Source**: Public medical databases, anonymized.
"""
    with open(os.path.join(cards_dir, 'dataset_card.md'), 'w') as f:
        f.write(dataset_card)

if __name__ == "__main__":
    generate_cards()
