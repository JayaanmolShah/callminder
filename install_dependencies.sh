#!/bin/bash

# Update and install system dependencies
apt-get update || true
apt-get install -y libportaudio2 || true
