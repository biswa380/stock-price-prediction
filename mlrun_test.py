# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:20:21 2022

@author: AMD
"""
from os import path
import mlrun

# Set the base project name
project_name_base = 'getting-started'

# Initialize the MLRun project object
project = mlrun.get_or_create_project(project_name_base, context="./", user_project=True)

# Display the current project name
project_name = project.metadata.name
print(f'Full project name: {project_name}')