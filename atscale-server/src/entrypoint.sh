#!/bin/sh

if [ -f ./first-time.set ]
then
  cp ./atscale.yaml /opt/atscale/conf
  chown atscale:atscale /opt/atscale/conf/atscale.yaml
  rm ./first-time.set
  su - atscale -c "/opt/atscale/versions/2021.3.0.3934/bin/configurator.sh --activate --automatic-install"
  curl --location --request PUT 'http://127.0.0.1:10502/license'  --header 'Content-Type: application/json' --data-binary '@/root/license/license.json'
  sleep 1s
  for file in /root/cubes/*
  do
  TOKEN=$(curl  -u admin:admin --location --request GET 'http://localhost:10500/default/auth')
  sleep 1s
  curl --location --request POST "http://localhost:10500/api/1.0/org/default/project" --header "Authorization: Bearer $TOKEN" --header "Content-Type: application/xml" --data-binary "@$file"
  sleep 5s
  done
  sleep 20s
  for file1 in /root/connections/*
  do
  TOKEN=$(curl  -u admin:admin --location --request GET 'http://localhost:10500/default/auth')
  sleep 1s
  curl --location --request POST "http://localhost:10502/connection-groups/orgId/default"  --header "Authorization: Bearer $TOKEN" --header "Content-Type: application/json"  --data-binary "@$file1"
  sleep 5s
  done
else
  su - atscale -c "/opt/atscale/bin/atscale_start"
fi

exec "$@"




