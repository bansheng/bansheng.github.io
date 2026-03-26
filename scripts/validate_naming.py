import os
import re
import sys

# 文章存放目录
POSTS_DIR = 'content/posts'

# 命名规范：3位数字序号 + 下划线 + 英文小写字母与下划线组合
# 例如：001_my_first_post.md
NAMING_PATTERN = re.compile(r'^\d{3}_[a-z0-9_]+\.(md|docx|pdf)$')

def validate_naming():
    if not os.path.exists(POSTS_DIR):
        print(f"Error: Directory {POSTS_DIR} not found.")
        return False

    files = [f for f in os.listdir(POSTS_DIR) if os.path.isfile(os.path.join(POSTS_DIR, f))]
    invalid_files = []

    for filename in files:
        if not NAMING_PATTERN.match(filename):
            invalid_files.append(filename)

    if invalid_files:
        print("Naming Convention Violation Found:")
        for f in invalid_files:
            print(f"  - {f}")
        print("\nRequired Format: XXX_description_name.ext (e.g., 001_my_post.md)")
        return False
    
    print("All files follow the naming convention.")
    return True

if __name__ == "__main__":
    success = validate_naming()
    if not success:
        sys.exit(1)
    sys.exit(0)
