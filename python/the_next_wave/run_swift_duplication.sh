#!/bin/bash
socat -u TCP-LISTEN:3002,fork - | tee >(socat - TCP:192.168.1.99:3005) >(socat - TCP:192.168.1.99:3004) >(socat - TCP:192.168.1.99:3003) >(socat - TCP:192.168.1.99:3001) > /dev/null
