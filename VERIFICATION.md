# Verification status

Prepared for Suhani16-K/CA_2 using the supplied Kaggle CSV.

Completed locally:

- Real Jupyter kernel executed every notebook cell successfully; outputs are retained.
- 10,000 source rows; 127 exact duplicate records removed before splitting.
- 7,898 training records; 1,975 holdout test records.
- Holdout R²: 0.9884301209927054.
- Holdout MAE: 1.6469703984255568.
- Holdout RMSE: 2.075066490148804.
- Example predicted Performance Index: 91.81.
- Saved model reload reproduces all holdout predictions.
- 10 API/model tests passed, including positive predictions, input validation and saved-model quality checks.
- Original dataset and source attribution are included.

Prepared but not yet verified as executed:

- Hosted Colab execution in Suhani's account (ready-to-run notebook included).
- Local Docker image/container execution for this project.
- Push to Suhani16-K/CA_2 and its GitHub Actions result. The current account has read access only, and collaborator access was declined. No push has been made and no green CI result is claimed for Suhani's repository.

After upload, the included workflow tests the API, builds Docker and calls the container's HTTP endpoints. Check the newest Actions run before claiming all eight steps are complete.
