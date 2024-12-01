#!/bin/bash

az containerapp up   -g tandic-aca-rg   -n tandic-aca-app   --ingress external   --target-port 80   --source .

