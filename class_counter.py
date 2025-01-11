#!/usr/bin/env python3
"""
Simple script to increment and track class count.
"""

def increment_count(counter_file: str = "class_counter.txt") -> int:
    """Read current count, increment it, and save back to file."""
    try:
        with open(counter_file, 'r') as f:
            count = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        count = 0
    
    count += 1
    
    with open(counter_file, 'w') as f:
        f.write(str(count))
    
    return count

def get_count(counter_file: str = "class_counter.txt") -> int:
    """Get current count from file."""
    try:
        with open(counter_file, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Track class count")
    parser.add_argument("--increment", action="store_true", help="Increment the counter")
    args = parser.parse_args()
    
    if args.increment:
        count = increment_count()
        print(f"Incremented count to: {count}")
    else:
        count = get_count()
        print(f"Current count: {count}")
