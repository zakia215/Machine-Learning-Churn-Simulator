#!/bin/bash

# Copy dataset to the namenode container
docker cp dataset namenode:/tmp

# Create the HDFS destination directory if it doesn't exist
docker exec -it namenode bash -c "hdfs dfs -mkdir -p /user/root"

# Upload the dataset to HDFS
docker exec -it namenode bash -c "hdfs dfs -put -p -f /tmp/dataset /user/root"
