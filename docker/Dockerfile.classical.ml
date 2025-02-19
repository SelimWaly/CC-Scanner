FROM intel/intel-optimized-ml:scikit-learn-2023.1.1-xgboost-1.7.5-pip-base as base

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --no-install-recommends openssh-server wget vim net-tools git-all htop python3-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -P '' && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    sed -i 's/#   Port 22/Port 12345/' /etc/ssh/ssh_config && \
    sed -i 's/#Port 22/Port 12345/' /etc/ssh/sshd_config

##Install packages
COPY ../classical-ml/requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm -rf requirements.txt

COPY ../classical-ml /fraud-detection/classical-ml
COPY ../configs/single-node /workspace/configs
COPY ../wrapper.sh /fraud-detection/wrapper.sh

WORKDIR /fraud-detection/classical-ml

CMD ["sh", "-c", "service ssh start; bash"]
