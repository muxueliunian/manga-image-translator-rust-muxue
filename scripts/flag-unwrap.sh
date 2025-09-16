#!/bin/bash

SEARCH_DIR="./crates"
find "$SEARCH_DIR" -type f -name "*.rs" ! -path "*/benches/*" ! -path "*/kumiko/*" | while read -r file; do
    prev_line=""
    line_number=0
    while IFS= read -r line; do
        ((line_number++))

        if [[ "$line" == *"unwrap()"* ]]; then
            if [[ "$prev_line" != *"allow:unwrap"* ]]; then
                col=$(awk -v a="$line" 'BEGIN{print index(a,"unwrap()")}')
                echo "$file:$line_number:$col: $line"
            fi
        fi

        prev_line="$line"
    done < "$file"
done
