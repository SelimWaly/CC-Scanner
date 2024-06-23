#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

filepath=~/.ssh/id_rsa.pub
user=$1
password=$2
host_file=$3

service ssh start

auto_ssh_copy_id() {
  expect -c "set timeout -1;
  spawn ssh-copy-id -i $4 $2@$1;
    expect {
      {Are you sure you want to continue connecting *} {send -- yes\r;exp_continue;}
      {*password:} {send -- $3\r;exp_continue;}
      eof {exit 0;}
    };"
}

# Determine whether the local public key exists, if not, you need to generate a public key
[ ! -f $filepath ] && {
  ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
}

hosts=`cat ${host_file} | grep -v "^#"`
# Perform secret-free login operation for each host in the configuration file
for serverIp in $hosts
do
  echo $serverIp--$user--$password
  auto_ssh_copy_id $serverIp $user $password $filepath
done
