#!/bin/bash

SEARCH_DIR="./crates"

# Find all .rs files except benches and kumiko directories
find "$SEARCH_DIR" -type f -name "*.rs" \
    ! -path "*/benches/*" \
    ! -path "*/kumiko/*" | while read -r file; do

    line_number=0
    while IFS= read -r line; do
        ((line_number++))
        if [[ "$line" == *"TODO"* || "$line" == *"todo"* ]]; then
            # Get column (first occurrence of TODO/todo)
            col=$(awk -v a="$line" 'BEGIN{print index(tolower(a),"todo")}')
            echo "$file:$line_number:$col: $line"
        fi
    done < "$file"
done
