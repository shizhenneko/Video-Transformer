import os

def modify_urls(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified_lines = []
    for i, line in enumerate(lines, 1):
        url = line.strip()
        if not url:
            modified_lines.append("")
            continue
        
        # Remove trailing & characters
        while url.endswith('&'):
            url = url[:-1]
            
        # Check if ? exists
        if '?' in url:
            new_url = f"{url}&p={i}"
        else:
            new_url = f"{url}?p={i}"
            
        modified_lines.append(new_url)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(modified_lines) + '\n')
    
    print(f"Successfully modified {len(modified_lines)} URLs in {file_path}")

if __name__ == "__main__":
    modify_urls('URL.txt')
