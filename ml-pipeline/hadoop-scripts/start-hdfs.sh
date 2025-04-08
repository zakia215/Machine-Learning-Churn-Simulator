#!/bin/bash
if [ ! -d "/opt/hadoop/data/nameNode/current" ]; then
    echo "Formatting NameNode..."
    chown -R airflow:airflow /opt/hadoop/data/nameNode
    chmod 755 /opt/hadoop/data/nameNode
    yes Y | hdfs namenode -format
fi
hdfs namenode