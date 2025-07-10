import os

def print_file_tree(root_path, max_files_per_level=30, prefix=""):
    """
    打印文件夹树结构，每层最多显示max_files_per_level个文件，多余用"..."隐藏。
    不显示隐藏文件（以.开头）。
    :param root_path: 根目录路径
    :param max_files_per_level: 每层最多显示的文件/文件夹数
    :param prefix: 前缀（递归用）
    """
    try:
        items = sorted([f for f in os.listdir(root_path) if not f.startswith('.')])
    except Exception as e:
        print(prefix + "[无法访问]")
        return

    count = 0
    for i, name in enumerate(items):
        if count >= max_files_per_level:
            print(prefix + "...")
            break
        path = os.path.join(root_path, name)
        connector = "├── " if i < len(items) - 1 else "└── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            new_prefix = prefix + ("│   " if i < len(items) - 1 else "    ")
            print_file_tree(path, max_files_per_level, new_prefix)
        count += 1

# 示例用法
if __name__ == "__main__":
    print_file_tree(".", max_files_per_level=30)