#! /bin/bash

# start sshd server
/usr/sbin/sshd

# start jupyter
jupyter notebook --allow-root --notebook-dir=/notebooks "$@"