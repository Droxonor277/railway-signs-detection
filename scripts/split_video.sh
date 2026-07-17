#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <video_file> [output_folder] <duration>"
    echo ""
    echo "  video_file    : path to the input video file"
    echo "  output_folder : (optional) directory for output chunks; defaults to same directory as input"
    echo "  duration      : chunk duration in HH:MM:SS format (e.g. 00:05:00 for 5-minute chunks)"
    echo ""
    echo "Output files are named: <original_name>-<index>-<hh>-<mm>-<ss>.<ext>"
    echo "where <index> is 1-based and <hh>-<mm>-<ss> is the start timestamp of the chunk."
    exit 1
}

# -- Argument parsing 
if [[ $# -eq 2 ]]; then
    input="$1"
    duration="$2"
    output_dir="$(dirname "$input")"
elif [[ $# -eq 3 ]]; then
    input="$1"
    output_dir="$2"
    duration="$3"
else
    usage
fi

# -- Validation 
if [[ ! -f "$input" ]]; then
    echo "Error: input file not found: $input" >&2
    exit 1
fi

if ! [[ "$duration" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Error: duration must be in HH:MM:SS format (e.g. 00:05:00)" >&2
    exit 1
fi

# -- Derived values --
mkdir -p "$output_dir"

filename="$(basename "$input")"
basename_noext="${filename%.*}"
ext="${filename##*.}"

# Total duration in whole seconds (floor)
total_seconds=$(ffprobe -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$input" | cut -d. -f1)

# Chunk duration in seconds
IFS=: read -r hh mm ss <<< "$duration"
chunk_seconds=$(( 10#$hh * 3600 + 10#$mm * 60 + 10#$ss ))

if [[ $chunk_seconds -le 0 ]]; then
    echo "Error: chunk duration must be greater than 00:00:00" >&2
    exit 1
fi

# Ceiling division to include any remainder as the last chunk
num_chunks=$(( (total_seconds + chunk_seconds - 1) / chunk_seconds ))

echo "Input        : $input"
echo "Total length : ${total_seconds}s"
echo "Chunk length : ${chunk_seconds}s  (${duration})"
echo "Chunks       : $num_chunks"
echo "Output dir   : $output_dir"
echo ""

# -- Splitting ─
for (( i=0; i<num_chunks; i++ )); do
    start=$(( i * chunk_seconds ))

    start_hh=$(( start / 3600 ))
    start_mm=$(( (start % 3600) / 60 ))
    start_ss=$(( start % 60 ))
    timestamp=$(printf "%02d-%02d-%02d" "$start_hh" "$start_mm" "$start_ss")

    index=$(( i + 1 ))
    output_file="${output_dir}/${basename_noext}-${index}-${timestamp}.${ext}"

    echo "  [$index/$num_chunks] start=${start}s → $output_file"

    ffmpeg -v warning \
        -ss "$start" \
        -i "$input" \
        -t "$chunk_seconds" \
        -c copy \
        "$output_file"
done

echo ""
echo "Done — $num_chunks chunk(s) written to '$output_dir'."
