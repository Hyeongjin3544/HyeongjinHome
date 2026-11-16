# Repository Workflow Rules

## 1. Pull Request Workflow (No Direct Push to `main`)

- **Never commit directly to `main` or push directly to `origin/main`**, unless the user explicitly asks to bypass the PR process.
- Pushing to `main` immediately triggers `.github/workflows/deploy.yml` to publish the live GitHub Pages site.
- Always work on `dev` (or a descriptive feature branch such as `notes/<topic>` or `fix/<issue>`) branched from the latest `origin/main`:
  1. Fetch the latest `origin` state (`git fetch origin`).
  2. Switch to `dev` (or create a new branch) synced from `origin/main` (e.g., `git checkout -B dev origin/main` when starting fresh after a merged PR).
  3. Make edits, verify links/HTML/math formatting locally, and commit changes to the working branch.
  4. Push the working branch to `origin` (`git push -u origin <branch>`).
  5. Do **not** merge into `main` yourself. Provide the GitHub Pull Request creation link (`https://github.com/Hyeongjin3544/HyeongjinHome/compare/main...<branch>?expand=1`, or open a PR with `gh pr create` if `gh` is installed) so the user can review and merge on GitHub.
- After the user merges a PR on GitHub, syncing the workspace means fetching `origin`, fast-forwarding `main` (`origin/main`), and keeping the working branch (`dev`) up to date.
