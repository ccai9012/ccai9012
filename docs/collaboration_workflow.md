# Collaboration Workflow

This page is for students and TAs who need to make a small, reviewable change
to the course repository. It covers the public workflow. Maintainer-only build
details stay in the local, non-public `docs/maintainer_authoring.md` handover.

## Before you begin

You need GitHub Desktop, a GitHub account with access to the repository, and a
local copy of the repository. Git records the history of edits; GitHub is the
shared copy that others can review. You do not need to memorise Git commands to
follow this workflow.

For a terminal refresher, open Terminal (macOS/Linux) or PowerShell (Windows)
and try:

```text
pwd                 # show the current folder
ls                  # list files (PowerShell: dir)
cd path/to/ccai9012 # move into the repository
```

If the course task needs Python, use the [Installation Guide](installation.html)
to create the `ccai9012` environment before running a notebook or build.

## Clone and open the repository

1. In GitHub Desktop, choose **File → Clone repository**.
2. Select the course repository and a local folder with enough space for the
   notebooks and sample data.
3. Choose **Clone**, then select **Open in your external editor**.
4. Confirm that the editor opens the repository root—the folder containing
   `environment.yml`, `ccai9012/`, `weekly_scripts/`, and `starter_kits/`.

## Edit and inspect

Make one focused change at a time. For a Markdown page, edit the source under
`docs/md/`; for a tutorial, edit the relevant notebook under
`weekly_scripts/`. Keep API keys, private pricing, and personal paths out of
the file.

Return to GitHub Desktop and open the **Changes** tab. Read the diff as a
reviewer would: check that the changed files are expected, that no notebook
checkpoint or generated cache was included, and that images and links point to
repository-local paths where possible.

## Commit a focused change

Write a short message that says what the change does, for example
`Clarify Week 3 data structures`. Select only the intended files and choose
**Commit to the current branch**. A commit is a review point, not a backup of
tokens or a place to store generated model files.

## Pull before push

Before pushing, choose **Fetch origin**, then **Pull origin**. This incorporates
new work from classmates or staff before your change is shared. Re-open the
**Changes** tab and check the result. If the branch has diverged, stop and
resolve the conflict before pushing.

## Resolve a simple conflict

GitHub Desktop lists conflicted files and offers **Open in external editor**.
Keep the intended parts from both versions, remove the conflict markers
(`<<<<<<<`, `=======`, `>>>>>>>`), save the file, and re-open the diff. For a
notebook conflict, do not guess at raw JSON: keep the safer version, ask a TA
to help merge the cells, or restore the notebook from a known good commit.
Commit the resolution only after the file opens and the relevant check passes.

If the conflict involves credentials, private material, a deleted teaching
asset, or a change you cannot explain, do not force-push. Save the state and
escalate to the course maintainer with the file name and the two competing
commits.

## Push and verify

Choose **Push origin** only after the pull and diff review succeed. On GitHub,
open the branch or pull request, inspect the changed-files tab, and check the
rendered Markdown or published page. Verify one notebook or build command
that is directly affected by the change. A green upload is not evidence that
the teaching content or links are correct.

## Small hand-off checklist

- [ ] The change has one clear purpose and a readable diff.
- [ ] I pulled before pushing and resolved conflicts explicitly.
- [ ] No token, private pricing, personal path, checkpoint, or large generated
      output is included.
- [ ] The relevant notebook, script, or documentation check passed.
- [ ] The GitHub branch or published page shows the intended result.
