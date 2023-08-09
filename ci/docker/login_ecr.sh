#!/bin/bash

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 650140442593.dkr.ecr.us-west-2.amazonaws.com
