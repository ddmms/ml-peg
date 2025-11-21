#!/usr/bin/env bash

version=$1

sed -i.bak "s;^version =.*;version = \"$version\";g" pyproject.toml && rm pyproject.toml.bak

git add pyproject.toml
git commit -m "Bump version for release $version"
git tag -f -a v$version -m "Release $version"
git push
git push --tags
