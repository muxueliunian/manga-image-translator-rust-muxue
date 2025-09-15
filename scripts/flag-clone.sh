#!/bin/bash

SEARCH_DIR="./crates"
find "$SEARCH_DIR" -type f -name "*.rs" ! -path "*/benches/*"  ! -path "*/kumiko/*" | while read -r file; do
    allow_clone=false
    line_number=0
    while IFS= read -r line; do
        ((line_number++))
        if [[ "$line" == *"allow:clone"* ]]; then
            allow_clone=true
        fi

        if [[ "$line" == *"clone()"* && "$allow_clone" == false ]]; then
            col=$(awk -v a="$line" 'BEGIN{print index(a,"clone()")}')
            echo "$file:$line_number:$col: $line"
        fi
    done < "$file"
done
