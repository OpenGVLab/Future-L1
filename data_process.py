import os
import json
import sys
import argparse

def process_single_file(file_path, prefix):
    """Process a single JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified = False
        
        # Unified processing function: supports dict and list
        def update_item(item):
            nonlocal modified
            for field in ["image", "reasoning_image"]:
                if field in item and isinstance(item[field], list):
                    # Add prefix to each path in the list
                    item[field] = [os.path.join(prefix, p) for p in item[field]]
                    modified = True

        if isinstance(data, list):
            for sample in data:
                update_item(sample)
        elif isinstance(data, dict):
            update_item(data)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch add prefix to image paths in JSON files")
    parser.add_argument("input_dir", type=str, help="Path to input directory (also used as the prefix)")
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist")
        return

    count = 0

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                # Use args.input_dir as the prefix
                if process_single_file(file_path, args.input_dir):
                    print(f"Updated: {file_path}")
                    count += 1

    print(f"\nProcessing complete! Modified {count} files in total.")

if __name__ == "__main__":
    main()
