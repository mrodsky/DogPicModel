#!/bin/zsh

# Ensure the target directory exists
mkdir -p ./dog_dataset/dogs

# Find and move all .jpeg and .jpg files from subfolders to the ./dog_dataset/dogs/ directory
find ./dog_dataset/ -type f \( -iname "*.jpeg" -o -iname "*.jpg" \) -exec mv {} ./dog_dataset/dogs/ \;

echo "All .jpeg and .jpg files have been moved to ./dog_dataset/dogs/"
