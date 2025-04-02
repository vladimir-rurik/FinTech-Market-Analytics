import os

def show_tree_and_py_content(path, indent_level=0):
    """
    Recursively walk through 'path', printing each directory and file.
    If a file is a Python file (.py), also print its contents.
    """
    # Create a visual indent for sub-items
    indent = '    ' * indent_level
    
    try:
        # Get all entries (files and directories) in the current path
        entries = os.listdir(path)
    except PermissionError:
        # If we don't have permission to list this directory, skip it
        print(f"{indent}[ACCESS DENIED] {path}")
        return

    # Sort entries so that directories and files appear in a stable order
    entries.sort()

    for entry in entries:
        full_path = os.path.join(path, entry)

        if os.path.isdir(full_path):
            # Print directory name
            print(f"{indent}+ {entry}/")
            # Recursively walk into the directory
            show_tree_and_py_content(full_path, indent_level + 1)
        else:
            # Print file name
            print(f"{indent}- {entry}")
            # If it is a .py file, read and display its content
            if entry.lower().endswith('.py'):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Print the file's content in the desired format
                    print(f"{indent}  filename: {full_path}\n{indent}  content:")
                    # Indent each line of the content for clarity
                    for line in content.splitlines():
                        print(f"{indent}    {line}")
                    print()  # Extra blank line for separation
                except Exception as e:
                    print(f"{indent}  [ERROR READING FILE] {e}")

if __name__ == "__main__":
    # You can specify any path you want here. By default, it uses the current working directory.
    start_path = os.getcwd()
    
    print(f"Showing directory tree for: {start_path}\n")
    show_tree_and_py_content(start_path)
