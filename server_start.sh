# cd ${SRC_PATH}/src/PythonFlask && ${PYTHONPATH} -m gunicorn --reload --bind "0.0.0.0:8080" app:app &

/home/irisowner/.local/bin/jupyter-notebook --no-browser --port=8888 --ip 0.0.0.0 --notebook-dir=/opt/irisapp/src/Notebooks --NotebookApp.token='' --NotebookApp.password='' &

exit 1