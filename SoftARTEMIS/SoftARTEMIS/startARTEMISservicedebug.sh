#!/bin/bash
sudo chmod 777 /dev/i2c-5

#sudo ufw disable
#sudo upnpc -r 319 udp 320 udp 319 tcp 320 tcp 8177 tcp 8177 udp 7 udp 7 tcp 22 tcp 22 udp
#sudo gnome-terminal -- bash -c "ptpd --slaveonly -u 192.168.0.102 --interface eth0 --verbose"

gnome-terminal -- bash -c "source /opt/aws/deepracer/setup.bash; python /home/deepracer/SoftARTEMIS/cloud_control_node_UDP.py; bash"

gnome-terminal -- bash -c "source /opt/aws/deepracer/setup.bash; python /home/deepracer/SoftARTEMIS/autonomous_control_node.py; bash"

gnome-terminal -- bash -c "source /opt/aws/deepracer/setup.bash; python /home/deepracer/SoftARTEMIS/admin_communications_node.py; bash"
