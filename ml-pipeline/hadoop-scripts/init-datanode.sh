#!/bin/bash
if [ ! -d "/opt/hadoop/data/dataNode/current" ]; then
    echo "Formatting dataNode..."
    chown -R airflow:airflow /opt/hadoop/data/dataNode
    chmod 755 /opt/hadoop/data/dataNode
    hdfs datanode -format
fi
rm -rf /opt/hadoop/data/dataNode/*
hdfs datanode