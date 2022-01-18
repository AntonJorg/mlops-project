export REGION=us-central1
export JOB_NAME=custom_container_job_cpu_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
  --scale-tier custom \
  --region $REGION \
  --master-image-uri gcr.io/dtumlops-detr/testing:latest \
  --master-machine-type e2-highmem-16 \
  -- default_root_dir=gs://dtumlops-detr-aiplatform \
  experiment.batch_size=1 \
  experiment.trainer.limit_val_batches=1 \
  experiment.trainer.limit_train_batches=1

