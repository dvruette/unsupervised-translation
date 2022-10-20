#!/bin/bash


LOCAL_PORT=8899
USER=$1
WAITING_TIME_INTERVAL=30

# Find old tunnel and delete it
pid=$(ps | grep ssh | grep $LOCAL_PORT | cut -d " " -f 1)
if [ ! -z "$pid" ]; then
  echo "Closing open tunnel on port $LOCAL_PORT"
  kill $pid
fi

ssh $USER "while ! [ -e \$HOME/VSCode_tunnel ]; do echo 'Waiting for code server to start, sleep for $WAITING_TIME_INTERVAL sec'; sleep $WAITING_TIME_INTERVAL; done"

remote_address=$(ssh $USER "cat \$HOME/VSCode_tunnel")
ssh -q -N -L localhost:$LOCAL_PORT:$remote_address $USER &

# SSH tunnel is started in the background, pause 5 seconds to make sure
# it is established before starting the browser
sleep 5

# save url in variable
appurl=http://localhost:$LOCAL_PORT
echo -e "Starting browser and connecting it to code server"
echo -e "Connecting to url "$appurl

APP=$(find "$HOME/Applications" -name "code-server.app" | head -n 1)
if [[ ! -z APP ]]; then
  open "$APP"
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
	xdg-open $appurl
elif [[ "$OSTYPE" == "darwin"* ]]; then
	open $appurl
else
	echo -e "Your operating system does not allow to start the browser automatically."
  echo -e "Please open $appurl in your browser."
fi