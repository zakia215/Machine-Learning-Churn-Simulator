FROM apache/airflow:2.10.4-python3.9

USER root
RUN apt-get update && apt-get install -y gcc libpq-dev procps default-jre wget curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV AIRFLOW_HOME=/opt/airflow
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

ENV HADOOP_VERSION=3.4.1
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/etc/hadoop
ENV MULTIHOMED_NETWORK=1

RUN HADOOP_URL="https://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
    && curl 'https://dist.apache.org/repos/dist/release/hadoop/common/KEYS' | gpg --import - \
    && curl -fSL "$HADOOP_URL" -o /tmp/hadoop.tar.gz \
    && curl -fSL "$HADOOP_URL.asc" -o /tmp/hadoop.tar.gz.asc \
    && gpg --verify /tmp/hadoop.tar.gz.asc \
    && mkdir -p "${HADOOP_HOME}" \
    && tar -xvf /tmp/hadoop.tar.gz -C "${HADOOP_HOME}" --strip-components=1 \
    && rm /tmp/hadoop.tar.gz /tmp/hadoop.tar.gz.asc \
    && ln -s "${HADOOP_HOME}/etc/hadoop" /etc/hadoop \
    && mkdir "${HADOOP_HOME}/logs" \
    && mkdir /hadoop-data

ENV SPARK_VERSION=3.5.3
ENV SPARK_HOME=/opt/spark
RUN wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3 ${SPARK_HOME} && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.tgz

ENV PATH="${SPARK_HOME}/bin:${HADOOP_HOME}/bin:${PATH}"

# TODO on CI/CD : Copy DAGs and scripts

USER airflow

RUN pip install --upgrade pip && \
pip install pandas numpy mlflow pyspark==${SPARK_VERSION} scikit-learn boto3 apache-airflow apache-airflow-providers-apache-spark 

# Set the entrypoint
WORKDIR $AIRFLOW_HOME