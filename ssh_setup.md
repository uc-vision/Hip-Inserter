# SSH Setup

 - External computer: Your computer
 - Gateway server: Cloud vps instance
 - Target computer: Doctors Laptop

## Setup Gateway Server

 1. Create a vps instance (such as a digital ocean droplet)
 2. Add the following to /etc/ssh/sshd_config
 ```
 GatewayPorts yes
 ```

## Setup Target Computer

### External Access

 1. Install the following on the target computer
 ```
 sudo apt install autossh openssh-server
 ```
 2. create ssh public/private keypairs
 ```
 ssh-keygen
 ```
 3. Add the following to ~/.ssh/config (Hostname is the gate server public ip)
 ```
 Host gateway
    HostName             170.64.189.106
    User                 root
    RemoteForward        2222 localhost:22
    ExitOnForwardFailure yes
    ServerAliveInterval  10
    ServerAliveCountMax  3
 ```
 4. Copy public key to gateway server.
 ```
 ssh-copy-id gateway
 ```
 5. ssh in once to accept certificate.
 ```
 ssh gateway
 ```
 6. With the ssh connection running on the target computer, you should be able to ssh in from an external computer, through the gateway server into the target computer. Note: `user` is the target computer username, `170.64.189.106` is the gateway server public ip, and `2222` is the exposed port on the gateway server.
 ```
 ssh user@170.64.189.106 -p 2222
 ```

### AutoSSH
Now we add a service that starts a connection on boot on the target computer, so that we can always connect to it.

 1. Create the following file.
 ```
 sudo nano /etc/systemd/system/gateway.service
 ```
 2. Add the following.
 ```
[Unit]
Description=AutoSSH gateway service
After=network.target

[Service]
User=user
Environment="AUTOSSH_GATETIME=0"
ExecStart=/usr/bin/autossh -M 0 -N gateway

[Install]
WantedBy=multi-user.target
 ```

 3. Enable systemd service
 ```
 sudo systemctl enable gateway
 ```

 4. Reboot target computer.
 5. Check if you can now externally access the target computer.
 6. (Optional) You can check on the status of the gateway service with the following.
 ```
 sudo systemctl status gateway -l
 ```