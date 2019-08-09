# GCP Notebook Example

## DM Deploy

```bash
./deploy.sh <project_id> <resource> <action>

# Create Example
./deploy.sh <project_id> iam create
./deploy.sh <project_id> iam notebook
```

### Upload

Run these from the root of the project (up one level)

```bash
gsutil -m rsync -d -r dataset gs://devopstar/projects/data-science/UGATIT/dataset
gsutil -m rsync -d -r samples gs://devopstar/projects/data-science/UGATIT/samples
gsutil -m rsync -d -r checkpoint gs://devopstar/projects/data-science/UGATIT/checkpoint

# Or copy zipped dataset from local
gsutil cp dataset/*.zip gs://devopstar/projects/data-science/UGATIT
```

### Download

Run these from the root of the project (up one level)

```bash
gsutil -m rsync -d -r gs://devopstar/projects/data-science/UGATIT/dataset ./dataset
gsutil -m rsync -d -r gs://devopstar/projects/data-science/UGATIT/samples ./samples
gsutil -m rsync -d -r gs://devopstar/projects/data-science/UGATIT/checkpoint ./checkpoint

# Or copy zipped dataset
gsutil cp gs://devopstar/projects/data-science/UGATIT/*.zip ./dataset
cd dataset
unzip selfie2anime.zip
```
