# DigitalOcean setup for this lab

This repo is prepared to run the lab on DigitalOcean:

- DigitalOcean Spaces replaces GCS/S3/Azure Blob.
- DigitalOcean Droplet replaces GCE/EC2/Azure VM.
- GitHub Actions still runs test, train, eval, and deploy.

## 1. Local Python

Use Python 3.10 for the lab.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
pytest tests/ -v
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_data.py
pytest tests/ -v
```

## 2. Create a DigitalOcean Space

Create a Space in the DigitalOcean dashboard. Recommended region for Vietnam:

```text
sgp1
```

Create a Spaces access key and save:

```text
SPACES_ACCESS_KEY
SPACES_SECRET_KEY
SPACES_REGION=sgp1
CLOUD_BUCKET=<your-space-name>
```

## 3. Configure DVC remote

Install dependencies first:

```bash
pip install -r requirements.txt
```

Configure DVC to use the Space:

```bash
dvc init
dvc remote add -d myremote s3://<your-space-name>/dvc
dvc remote modify myremote endpointurl https://sgp1.digitaloceanspaces.com
dvc remote modify myremote region sgp1
dvc remote modify --local myremote access_key_id <SPACES_ACCESS_KEY>
dvc remote modify --local myremote secret_access_key <SPACES_SECRET_KEY>
```

Track and upload data:

```bash
python generate_data.py
dvc add data/train_phase1.csv data/eval.csv data/train_phase2.csv
dvc push
git add .dvc/config data/*.dvc .gitignore
git commit -m "feat: track datasets with DVC on DigitalOcean Spaces"
```

Do not commit `.dvc/config.local`; it contains secrets.

## 4. Create a Droplet

Create an Ubuntu 22.04 Droplet. A basic shared CPU Droplet is enough for this lab.

Open inbound TCP:

```text
22    SSH
8000  FastAPI inference
```

SSH into the Droplet and install runtime dependencies:

```bash
sudo apt update
sudo apt install -y python3-pip curl
pip3 install fastapi==0.111.0 uvicorn==0.29.0 scikit-learn==1.4.2 joblib==1.4.2 boto3==1.43.0 pydantic
mkdir -p ~/src ~/models
```

Create the systemd service. This lab uses `root` as the Droplet login, so the service paths are under `/root`.

```bash
sudo tee /etc/systemd/system/mlops-serve.service > /dev/null <<EOF
[Unit]
Description=MLOps Model Inference Server
After=network.target

[Service]
User=root
WorkingDirectory=/root
Environment="SPACES_BUCKET=<your-space-name>"
Environment="SPACES_REGION=sgp1"
Environment="SPACES_ENDPOINT_URL=https://sgp1.digitaloceanspaces.com"
Environment="AWS_ACCESS_KEY_ID=<SPACES_ACCESS_KEY>"
Environment="AWS_SECRET_ACCESS_KEY=<SPACES_SECRET_KEY>"
ExecStart=/usr/bin/python3 /root/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlops-serve
```

The service can only start after the first successful pipeline uploads `models/latest/model.pkl`.

## 5. Deploy SSH key for GitHub Actions

On your local machine:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/mlops_deploy -N "" -C "github-actions-deploy"
```

Add the public key to the Droplet:

```bash
ssh <vm-user>@<vm-ip> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys" < ~/.ssh/mlops_deploy.pub
```

## 6. GitHub Actions secrets

Add these repository secrets:

```text
SPACES_ACCESS_KEY      DigitalOcean Spaces access key
SPACES_SECRET_KEY      DigitalOcean Spaces secret key
SPACES_REGION          sgp1
CLOUD_BUCKET           your Space name
VM_HOST                public IP of the Droplet
VM_USER                Linux user on the Droplet
VM_SSH_KEY             contents of ~/.ssh/mlops_deploy private key
```

## 7. First pipeline run

Push to `main`, or run the workflow manually from the GitHub Actions tab.

After the first successful run, confirm:

```bash
curl http://<vm-ip>:8000/health
curl -X POST http://<vm-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```
