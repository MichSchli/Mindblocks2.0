#!/bin/bash

pipenv run python setup.py clean
pipenv run python setup.py sdist bdist_wheel