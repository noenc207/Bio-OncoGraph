import os
import zipfile

def main():
    zip_filename = "benchmark_reproducibility.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists("results"):
            for root, dirs, files in os.walk("results"):
                for file in files:
                    zipf.write(os.path.join(root, file))
                
        if os.path.exists("scripts/benchmarking"):
            for root, dirs, files in os.walk("scripts/benchmarking"):
                for file in files:
                    if file.endswith(".py"):
                        zipf.write(os.path.join(root, file))
                    
    print(f"Packaged everything into {zip_filename}")

if __name__ == "__main__":
    main()
