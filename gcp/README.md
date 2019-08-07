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
gsutil -m cp checkpoint/*/UGATIT_light.model-13000.* gs://devopstar/projects/data-science/UGATIT/checkpoint
```

### Download

Run these from the root of the project (up one level)

```bash
gsutil -m rsync -d -r gs://devopstar/projects/data-science/UGATIT/dataset ./dataset
gsutil -m rsync -d -r gs://devopstar/projects/data-science/UGATIT/samples ./samples
gsutil -m rsync -d -r gs://devopstar/projects/data-science/UGATIT/checkpoint ./checkpoint
```
