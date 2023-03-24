#!/bin/bash

# Cleanup
rm -rf wiki/
rm -rf cachyos-website/
rm -rf arch-wiki/

# Fetch input data
git clone https://github.com/CachyOS/wiki
rm -rf wiki/*{png,svg}

mkdir arch-wiki
cp -r /usr/share/doc/arch-wiki/html/en/* arch-wiki

mkdir cachyos-website
curl -sSL https://cachyos.org > cachyos-website/main-page.html
curl -sSL https://cachyos.org/about > cachyos-website/about-page.html
curl -sSL https://cachyos.org/download > cachyos-website/download-page.html

# Install required python packages
pipenv install
