#!/bin/bash

az acr build --platform linux/amd64 \
    -t cada83652972acr.azurecr.io/tandic-aca:latest \
    -r cada83652972acr.azurecr.io .

az containerapp update --name tandic-aca-app  \
  --resource-group tandic-aca-rg \
  --image cada83652972acr.azurecr.io/tandic-aca:latest