import os
import sys
from datetime import datetime

POSTS_DIR = 'content/posts'

def get_next_index():
    if not os.path.exists(POSTS_DIR):
        return 1
    files = [f for f in os.listdir(POSTS_DIR) if os.path.isfile(os.path.join(POSTS_DIR, f))]
    indices = []
    for f in files:
        if f[:3].isdigit():
            indices.append(int(f[:3]))
    return max(indices, default=0) + 1

def create_post(name):
    index = get_next_index()
    # 将名称转换为英文小写字母+下划线
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    filename = f"{index:03d}_{safe_name}.md"
    filepath = os.path.join(POSTS_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Error: File {filename} already exists.")
        return

    content = f"""---
title: "{name}"
date: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S+08:00')}
draft: false
tags: []
categories: []
---

在这里开始编写你的内容...
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created new post: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/new_post.py \"Descriptive Name\"")
        sys.exit(1)
    create_post(sys.argv[1])
