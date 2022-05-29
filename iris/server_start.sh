${PYTHONPATH} /irisdev/app/src/PythonFlask/app.py &

/home/irisowner/.local/bin/jupyter-notebook --no-browser --port=8888 --ip 0.0.0.0 --notebook-dir=/opt/irisapp/src/Notebooks --NotebookApp.token='' --NotebookApp.password='' &

exit 1