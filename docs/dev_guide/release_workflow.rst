.. _release_workflow:

Release workflow
================

This document will go through the steps of releasing Hydromt-Wflow to Pypi and conda.

1. Create a release branch
--------------------------
Create a release branch locally with an appropriate name

2. Bump the version number
--------------------------
Change the version number in the __init__.py file of the hydromt_wflow module

3. Update the changelog
-----------------------
Update the changelog header with the version of the release and include the date of release in brackets.
Also add extra information on the highlights of the release.

4. Check the dependencies
-------------------------
Update or newly install the default pixi environment and run the tests to check if everything is working with the latest versions of the dependencies.

5.
