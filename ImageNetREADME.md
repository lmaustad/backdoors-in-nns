Download a subset from imagenet from the following kaggle https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000

```
#!/bin/bash
curl -L -o ~/BackdoorMaster/Datasets/imagenetmini-1000.zip\
  https://www.kaggle.com/api/v1/datasets/download/ifigotin/imagenetmini-1000
```

First create a kaggle account and get your (legacy) API key from https://www.kaggle.com/general. Then, place the downloaded `kaggle.json` file in `~/.kaggle/` (you may need to create this directory). Update its permissions: 

```
chmod 600 ~/.kaggle/kaggle.json
````

 Finally, run the top command to download the dataset.