
#!/bin/sh

PROJECT_ID="$1"

# Set Project
gcloud config set project $PROJECT_ID

case "$3" in
    "create"    )
        DEPLOY_ACTION="create"
        ;;
    "delete"    )
        DEPLOY_ACTION="delete"
        ;;
    *           )
        echo "Script requires an action. E.g. create, delete"
        exit 1
        ;;
esac

case "$2" in
    "iam"           )
        PROJECT_NUM=$(gcloud projects list \
            --filter=PROJECT_ID=$PROJECT_ID \
            --format="value(PROJECT_NUMBER)")
        if [ "$DEPLOY_ACTION" = "create" ]; then
            gcloud projects add-iam-policy-binding $PROJECT_ID \
                --member serviceAccount:$PROJECT_NUM@cloudservices.gserviceaccount.com  \
                --role roles/owner
        else
            echo "Deleting $PROJECT_ID-iam"
            gcloud projects remove-iam-policy-binding $PROJECT_ID \
                --member serviceAccount:$PROJECT_NUM@cloudservices.gserviceaccount.com  \
                --role roles/owner
        fi
        ;;
    "notebook"       )
        if [ "$DEPLOY_ACTION" = "create" ]; then
            gcloud deployment-manager deployments create $PROJECT_ID-notebook \
                --config resources/notebook_instance.yml
        else
            echo "Deleting $PROJECT_ID-notebook"
            gcloud deployment-manager deployments delete $PROJECT_ID-notebook
        fi
        ;;
    *               )
        echo "Script requires a resource. E.g. iam, notebook"
        exit 1
        ;;
esac