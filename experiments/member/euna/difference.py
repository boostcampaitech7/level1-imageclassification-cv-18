import pandas as pd

def compare_csv_files(file1, file2):
    # Load the CSV files
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    
    differences = []

    # Assuming both files have the same structure: 'ID', 'image_path', and 'target'
    for index, (row1, row2) in enumerate(zip(data1.itertuples(), data2.itertuples()), start=1):
        if row1.target != row2.target:
            differences.append({
                'index': index,
                'id': row1.ID,
                'image_path1': row1.image_path,
                'image_path2': row2.image_path,
                'target1': row1.target,
                'target2': row2.target
            })
    
    return differences

# Example file paths (replace with actual file paths)
file1 = '/mnt/data/file1.csv'
file2 = '/mnt/data/file2.csv'

diff_list = compare_csv_files(file1, file2)

# Print the differences
for diff in diff_list:
    print(f"Index: {diff['index']}")
    print(f"ID: {diff['id']}")
    print(f"Image Path 1: {diff['image_path1']} - Target 1: {diff['target1']}")
    print(f"Image Path 2: {diff['image_path2']} - Target 2: {diff['target2']}")
    print('-' * 50)

print(f"Total differences: {len(diff_list)}")