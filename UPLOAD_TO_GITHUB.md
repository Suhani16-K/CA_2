# Upload to Suhani's account

Repository: https://github.com/Suhani16-K/CA_2

1. Extract `Suhani_CA_2_Student_Performance.zip`.
2. Sign in as **Suhani16-K** and open the repository.
3. Choose **Add file → Upload files**.
4. Drag the extracted folder's contents into GitHub, including `.github`, `.vscode`, `app`, `artifacts`, `data`, `notebooks`, `scripts`, `tests` and the root files. Upload the contents, not the ZIP or an extra enclosing folder.
5. Ensure `.github/workflows/ci.yml` is included. Commit to `main` with message `Add Student Performance ML workflow`.
6. Open **Actions → Student Performance CI → test-and-docker** and wait for the green check. Enable Actions if GitHub displays an enable prompt.
7. Run the notebook in Colab (Copy to Drive → Runtime → Run all). Extract the downloaded `student_model_artifacts.zip` into `artifacts`, replacing its three files.
8. Download the executed Colab notebook using File → Download → Download .ipynb and replace the local notebook. Upload the changed model files and notebook into the same GitHub paths. CI runs again; use the newest successful run for submission.

No collaborator invitation or shared password/token is required. If uploads are denied, check that you are signed in as Suhani16-K.

Form answers:

- Dataset name: Student Performance (Multiple Linear Regression)
- Dataset source: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
- GitHub repository: https://github.com/Suhani16-K/CA_2
- Steps completed: claim 8/8 only after your notebook execution, API/Docker checks and GitHub Actions have succeeded.
