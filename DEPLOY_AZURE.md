# Deploy to Azure Container Apps (Azure for Students)

Portal menu names change, so look for the closest label. Tip: use "Desktop site" in your phone browser.

## Repo structure (Dockerfile at the root)
object-counting/
  Dockerfile
  app.py
  index.html
  requirements.txt
  README.md
  DEPLOY_AZURE.md
  notebooks/   (Colab notebook)
  docs/        (architecture diagram, test cases, report)
  screenshots/

## 1. Activate Azure for Students
1. Go to azure.microsoft.com/free/students and verify with your UCP student email. No credit card is needed.
2. Open portal.azure.com and check the subscription is "Azure for Students".

## 2. Put the code on GitHub
Upload Dockerfile, app.py, index.html and requirements.txt to the root of a GitHub repo.

## 3. Create the container app
1. Portal: search "Container Apps" > Create.
2. Subscription: Azure for Students. Resource group: create new (for example counting-rg).
3. Name: object-counting. Region: pick one the portal allows (student subscriptions limit regions).
4. Environment: create new, Consumption plan.
5. Finish creating with the sample container if it asks for an image (you replace it next).

## 4. Connect GitHub (builds your Dockerfile)
1. Open the container app > Settings > Continuous deployment.
2. Sign in to GitHub, choose your repo and branch (main).
3. Registry: create a new Azure Container Registry when offered. Dockerfile path: ./Dockerfile.
4. Save. GitHub Actions builds and deploys it (about 10 to 15 minutes the first time). Watch the repo's Actions tab.

## 5. Settings to change
- Ingress: enabled, accepting traffic from anywhere, target port 8000.
- Containers > Edit: CPU 2, Memory 4 Gi.
- Scale: minimum replicas 0, maximum replicas 1.

## 6. Test and demo
- Open the app URL from the Overview page.
- The first request after idle is slow (the model loads). Open it 5 minutes before the demo and run one image.
- Results live in temporary storage and reset when the app restarts. Download the CSVs after use.

## 7. After the demo
Delete the resource group (Resource groups > counting-rg > Delete) so nothing keeps using your credit.

## Fallback: one command in Azure Cloud Shell
az containerapp up --name object-counting --resource-group counting-rg --location <allowed-region> --repo https://github.com/<you>/<repo> --ingress external --target-port 8000
az containerapp update --name object-counting --resource-group counting-rg --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1
(Check flags with --help if a command is rejected.)

## Troubleshooting
- Region not allowed: choose one from the list the portal shows.
- "Resource provider not registered": Subscriptions > Resource providers > register Microsoft.App, Microsoft.OperationalInsights, Microsoft.ContainerRegistry.
- Build fails: open the failed run in the GitHub Actions tab and send me the last 20 lines.
- App crashes or restarts: raise memory (for example 4 CPU / 8 Gi).
