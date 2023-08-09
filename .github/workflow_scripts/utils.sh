#!/bin/bash

# By default, all builds are cached
DISABLE_CACHE=false  # Eg. 'true' or 'false'


# Function to measure command execution time
measure_command_time() {
    local command="$1"

    # Start timing
    local start_time=$(date +%s)

    # Run the command
    eval "$command"

    # Calculate the time taken
    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))

    # Format the elapsed time for display
    local formatted_time=$(printf "%02dhr %02dmin %02dsec" $((elapsed_time / 3600)) $(((elapsed_time % 3600) / 60)) $((elapsed_time % 60)))

    # Print the elapsed time
    echo "Time taken for $command: $formatted_time"
}
