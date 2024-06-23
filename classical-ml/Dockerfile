# Copyright [2022-23] [Intel Corporation]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM intel/intel-optimized-ml:scikit-learn-2023.1.1-xgboost-1.7.5-pip-base as base


RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --no-install-recommends openssh-server net-tools git-all python3-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -P '' && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    sed -i 's/#   Port 22/Port 12345/' /etc/ssh/ssh_config && \
    sed -i 's/#Port 22/Port 12345/' /etc/ssh/sshd_config

##Install packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm -rf requirements.txt

WORKDIR /workspace

CMD ["sh", "-c", "service ssh start; bash"]
