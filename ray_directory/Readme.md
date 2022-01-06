# How to use ray tune to do experiments



At this time, there is an error when running experiments related to unpickle in cluster nodes. Need to be fixed. 

The error is just like this https://github.com/ray-project/ray/issues/11991

## Steps

If you have not installed Distiller, first

```shell
pip install -e ../.
```

Then 

```shell
ray submit PATH_TO_CLUSTER.yaml bash run.sh
```

