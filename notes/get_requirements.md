# How to Get Requirements for a Project

You must manually track which packages are needed.
This command only helps you get the version of each
listed package.

```pip list --format=freeze | grep -E '^(ipython|numpy|pandas|pyarrow)=='```
