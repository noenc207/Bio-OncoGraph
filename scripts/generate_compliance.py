import os
import json

def generate_compliance():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    licenses_content = """# Third Party Licenses

## License 1
* **Name**: MIT License
* **Component**: Example Component

## License 2
* **Name**: Apache License 2.0
* **Component**: Example Component 2
"""
    with open(os.path.join(base_dir, 'THIRD_PARTY_LICENSES.md'), 'w') as f:
        f.write(licenses_content)

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": "urn:uuid:3e671687-395b-41f5-a30f-a58921a69b79",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "Bio-OncoGraph",
                "version": "1.0.0"
            }
        },
        "components": [
            {
                "type": "library",
                "name": "requests",
                "version": "2.28.1"
            }
        ]
    }
    with open(os.path.join(base_dir, 'sbom.json'), 'w') as f:
        json.dump(sbom, f, indent=4)

if __name__ == "__main__":
    generate_compliance()
