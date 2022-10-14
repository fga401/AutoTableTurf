#!/bin/bash
cd "$(dirname "$0")" || exit 1
target="~/nxbt_server/"
host="pi"
echo "target_path: $target"
ssh $host "sudo rm -rf $target && mkdir -p $target"
scp -r * "$host:$target"