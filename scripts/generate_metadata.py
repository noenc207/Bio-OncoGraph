import os
import json

def generate_metadata():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    metadata_dir = os.path.join(base_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    zenodo_meta = {
        "metadata": {
            "title": "Bio-OncoGraph",
            "upload_type": "software",
            "description": "A comprehensive tool for oncology graph processing.",
            "creators": [{"name": "Bio-OncoGraph Team"}]
        }
    }
    with open(os.path.join(metadata_dir, 'zenodo.json'), 'w') as f:
        json.dump(zenodo_meta, f, indent=4)

    hf_meta = {
        "modelId": "bio-oncograph-team/model-a",
        "tags": ["biology", "oncology", "graphs"]
    }
    with open(os.path.join(metadata_dir, 'huggingface.json'), 'w') as f:
        json.dump(hf_meta, f, indent=4)
        
    swh_meta = {
        "name": "Bio-OncoGraph",
        "description": "Oncology graph processing",
        "url": "https://github.com/bio-oncograph/bio-oncograph"
    }
    with open(os.path.join(metadata_dir, 'software_heritage.json'), 'w') as f:
        json.dump(swh_meta, f, indent=4)

    doi_meta = {
        "data": {
            "type": "dois",
            "attributes": {
                "doi": "10.1234/bio-oncograph.v1",
                "prefix": "10.1234",
                "suffix": "bio-oncograph.v1",
                "url": "https://example.com/bio-oncograph"
            }
        }
    }
    with open(os.path.join(metadata_dir, 'doi.json'), 'w') as f:
        json.dump(doi_meta, f, indent=4)

if __name__ == "__main__":
    generate_metadata()
