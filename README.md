# IRIS-python-dashboards

This app shows data visualization. For example, Covid19 data was used. The Dash framework used to build Dashboards is a Python framework created by plotly for building interactive web applications. Dash is open source and building an application using this framework is viewed in a web browser.

## Installation

### Docker
The repo is dockerised so you can  clone/git pull the repo into any local directory

```
$ git clone https://github.com/NjekTt/iris-python-dashboards.git
```

Open the terminal in this directory and run:

```
$ docker-compose up -d
```

## App

After installation open http://localhost:8080/

After you will be redirected to the authorization page. Authorization is based on the IRIS database. Default login and password is _SYSTEM : SYS

![image](https://user-images.githubusercontent.com/47400570/155238798-4c1ca3b2-e0b9-4ffd-a934-5922f10bb0d5.png)

The main page will show the first dashboard, which visualizes Covid data taken from the local IRIS database

On the left is the navigation menu.

- Overview - shows the general dashboard
- Timeline - a map is shown, there is a player at the bottom of the map, timeline shows the dynamics of data on the world map
- IRIS python usage - guide how python embedded was used, how data was retrieved from the IRIS database and a small example of using the IRIS Native API for Python

Online demo link : http://atscale.teccod.ru:8080/

# Screencast

## Overview

![image](https://user-images.githubusercontent.com/47400570/155239138-0f614bb0-1fc1-4e19-9b2a-553bc56c2112.png)

## Timeline

![image](https://user-images.githubusercontent.com/47400570/155239097-a49ab95a-2b03-4170-a59b-8a5616be6962.png)

## IRIS python usage

![image](https://user-images.githubusercontent.com/47400570/155239038-99890fe8-ed9d-4a82-bfbe-21bf3d4cd80b.png)


